package tui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/ssh"
	gossh "golang.org/x/crypto/ssh"

	"github.com/guyreading/container-rl-ssh/internal/auth"
	"github.com/guyreading/container-rl-ssh/internal/client"
	"github.com/guyreading/container-rl-ssh/internal/db"
)

type RootModel struct {
	keys     *db.KeyStore
	gameAddr string
	model    tea.Model
}

func NewRootModel(keys *db.KeyStore, gameAddr string, s ssh.Session) (tea.Model, []tea.ProgramOption) {
	opts := []tea.ProgramOption{
		tea.WithInput(s),
		tea.WithOutput(s),
		tea.WithAltScreen(),
	}

	rm := &RootModel{
		keys:     keys,
		gameAddr: gameAddr,
	}

	pk := s.PublicKey()
	if pk != nil {
		encoded := strings.TrimSpace(string(gossh.MarshalAuthorizedKey(pk)))
		fingerprint := auth.Fingerprint(pk)

		entry := keys.Lookup(encoded)
		if entry != nil {
			rm.model = NewMenuModel(rm, entry.PlayerName)
		} else {
			rm.model = NewRegisterModel(rm, encoded, fingerprint)
		}
	} else {
		rm.model = NewRegisterModel(rm, "", "")
	}

	return rm, opts
}

func (m *RootModel) Init() tea.Cmd {
	return m.model.Init()
}

func (m *RootModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.model.Update(msg)
	m.model = next
	return m, cmd
}

func (m *RootModel) View() string {
	return m.model.View()
}

type SharedState struct {
	PlayerName string
	TCPClient  *client.GameClient
	GameID     int64
	PlayerIdx  int
	GameCode   string
	NumPlayers int
	NumColors  int
}
