package server

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/charmbracelet/log"
	"github.com/charmbracelet/wish"

	"github.com/guyreading/container-rl-ssh/internal/auth"
	"github.com/guyreading/container-rl-ssh/internal/db"
)

type Config struct {
	ListenAddr string
	HostKeyDir string
	KeysPath   string
	GameAddr   string
}

func Run(cfg Config) error {
	keys, err := db.NewKeyStore(cfg.KeysPath)
	if err != nil {
		return fmt.Errorf("open key store: %w", err)
	}

	if err := os.MkdirAll(cfg.HostKeyDir, 0700); err != nil {
		return fmt.Errorf("create host key dir: %w", err)
	}

	hostKeyPath := filepath.Join(cfg.HostKeyDir, "ssh_host_ed25519")
	if _, err := os.Stat(hostKeyPath); os.IsNotExist(err) {
		log.Info("Generating host key", "path", hostKeyPath)
	}

	authHandlers := &auth.Handlers{Keys: keys}

	srv, err := wish.NewServer(
		wish.WithAddress(cfg.ListenAddr),
		wish.WithHostKeyPath(hostKeyPath),
		wish.WithPublicKeyAuth(authHandlers.PublicKeyHandler),
		wish.WithPasswordAuth(authHandlers.PasswordHandler),
		wish.WithMiddleware(
			pythonMiddleware(keys, cfg.GameAddr),
		),
	)
	if err != nil {
		return fmt.Errorf("create server: %w", err)
	}

	log.Info("SSH server starting", "addr", cfg.ListenAddr)
	if err := srv.ListenAndServe(); err != nil {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}
