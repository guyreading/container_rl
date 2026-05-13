package server

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/charmbracelet/ssh"
	"github.com/charmbracelet/wish"
	"github.com/creack/pty"

	"github.com/guyreading/container-rl-ssh/internal/auth"
	"github.com/guyreading/container-rl-ssh/internal/db"
)

func pythonMiddleware(keys *db.KeyStore, gameAddr string) wish.Middleware {
	return func(next ssh.Handler) ssh.Handler {
		return func(s ssh.Session) {
			defer next(s)

			ctx := s.Context()
			authStatus, _ := ctx.Value(auth.CtxAuthStatus).(auth.AuthStatus)

			var playerName string

			switch authStatus {
			case auth.StatusKnown:
				playerName, _ = ctx.Value(auth.CtxPlayerName).(string)
			case auth.StatusNewKey:
				encoded, _ := ctx.Value(auth.CtxPublicKey).(string)
				fingerprint, _ := ctx.Value(auth.CtxFingerprint).(string)
				var ok bool
				playerName, ok = registerPlayerInteractive(s, keys, encoded, fingerprint)
				if !ok {
					return
				}
			case auth.StatusAnonymous, "":
				io.WriteString(s, "\r\n")
				io.WriteString(s, "This server requires an SSH key to play.\r\n")
				io.WriteString(s, "Run: ssh-keygen -t ed25519\r\n")
				io.WriteString(s, "Then reconnect with your key.\r\n\r\n")
				return
			default:
				io.WriteString(s, "Authentication failed.\r\n")
				return
			}

			if playerName == "" {
				io.WriteString(s, "Could not determine player name.\r\n")
				return
			}

			ptyReq, winCh, isPty := s.Pty()
			if !isPty {
				io.WriteString(s, "Terminal required. Use an SSH client with PTY support.\r\n")
				s.Exit(1)
				return
			}

			host, port := parseAddr(gameAddr)
			venvPython := findPython()
			cmd := exec.Command(venvPython, "-m", "container_rl.client.tui",
				"--host", host,
				"--port", port,
				"--player-name", playerName,
			)
			cmd.Env = append(os.Environ(),
				"TERM="+ptyReq.Term,
				"PYTHONUNBUFFERED=1",
			)

			f, err := pty.Start(cmd)
			if err != nil {
				fmt.Fprintf(s, "\r\nFailed to start game: %s\r\n", err)
				return
			}
			defer f.Close()

			setWinsize(f, ptyReq.Window.Width, ptyReq.Window.Height)

			go func() {
				for win := range winCh {
					setWinsize(f, win.Width, win.Height)
				}
			}()

			go io.Copy(f, s)
			io.Copy(s, f)

			cmd.Wait()
		}
	}
}

func setWinsize(f *os.File, w, h int) {
	syscall.Syscall(
		syscall.SYS_IOCTL,
		f.Fd(),
		uintptr(syscall.TIOCSWINSZ),
		uintptr(unsafe.Pointer(&struct{ h, w, x, y uint16 }{
			uint16(h), uint16(w), 0, 0,
		})),
	)
}

func registerPlayerInteractive(s ssh.Session, keys *db.KeyStore, encoded, fingerprint string) (string, bool) {
	io.WriteString(s, "\r\n")
	io.WriteString(s, "Welcome! Your SSH key is not yet registered.\r\n")
	io.WriteString(s, fmt.Sprintf("Key fingerprint: %s\r\n\r\n", fingerprint))

	for {
		username, err := readLine(s, "Choose a username: ")
		if err != nil {
			return "", false
		}
		if username == "" {
			io.WriteString(s, "Username cannot be empty.\r\n")
			continue
		}
		if len(username) < 2 {
			io.WriteString(s, "Username must be at least 2 characters.\r\n")
			continue
		}
		if len(username) > 16 {
			io.WriteString(s, "Username must be 16 characters or fewer.\r\n")
			continue
		}
		if existing := keys.GetByPlayerName(username); existing != nil {
			io.WriteString(s, fmt.Sprintf("'%s' is already taken.\r\n", username))
			continue
		}

		playerID := keys.NextPlayerID()
		if err := keys.Register(&db.KeyEntry{
			PlayerName:  username,
			PlayerID:    playerID,
			PublicKey:   encoded,
			Fingerprint: fingerprint,
			CreatedAt:   nowRfc3339(),
		}); err != nil {
			fmt.Fprintf(s, "Registration error: %s\r\n", err)
			return "", false
		}

		io.WriteString(s, fmt.Sprintf("Registered as '%s'! Connecting to game...\r\n\r\n", username))
		return username, true
	}
}

func readLine(s ssh.Session, prompt string) (string, error) {
	io.WriteString(s, prompt)
	var line []byte
	buf := make([]byte, 1)
	for {
		_, err := s.Read(buf)
		if err != nil {
			return "", err
		}
		b := buf[0]
		if b == '\r' || b == '\n' {
			io.WriteString(s, "\r\n")
			return strings.TrimSpace(string(line)), nil
		}
		if b == 127 || b == 8 {
			if len(line) > 0 {
				line = line[:len(line)-1]
				io.WriteString(s, "\b \b")
			}
			continue
		}
		if b >= 32 {
			line = append(line, b)
			s.Write(buf)
		}
	}
}

func nowRfc3339() string {
	return time.Now().UTC().Format(time.RFC3339)
}

func parseAddr(addr string) (host, port string) {
	parts := strings.SplitN(addr, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return addr, "9876"
}

func findPython() string {
	locations := []string{
		"/opt/container-rl/.venv/bin/python3",
		"/opt/container-rl/.venv/bin/python",
		"/usr/bin/python3",
		"/usr/bin/python",
	}
	for _, p := range locations {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "/usr/bin/python3"
}
