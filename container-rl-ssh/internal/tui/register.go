package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	"github.com/guyreading/container-rl-ssh/internal/db"
)

type RegisterModel struct {
	root        *RootModel
	publicKey   string
	fingerprint string
	hasKey      bool
	input       textinput.Model
	errorMsg    string
	quitting    bool
}

func NewRegisterModel(root *RootModel, publicKey, fingerprint string) *RegisterModel {
	hasKey := publicKey != ""
	m := &RegisterModel{
		root:        root,
		publicKey:   publicKey,
		fingerprint: fingerprint,
		hasKey:      hasKey,
	}

	if hasKey {
		t := textinput.New()
		t.Placeholder = "Choose a username"
		t.CharLimit = 16
		t.Focus()
		m.input = t
	}

	return m
}

func (m *RegisterModel) Init() tea.Cmd {
	if m.hasKey {
		return textinput.Blink
	}
	return nil
}

func (m *RegisterModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEsc:
			return m, tea.Quit
		case tea.KeyEnter:
			if m.hasKey {
				return m.handleRegister()
			}
			return m, tea.Quit
		}
	}

	if m.hasKey {
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m *RegisterModel) handleRegister() (tea.Model, tea.Cmd) {
	username := strings.TrimSpace(m.input.Value())
	if username == "" {
		m.errorMsg = "Username cannot be empty."
		return m, nil
	}
	if len(username) < 2 {
		m.errorMsg = "Username must be at least 2 characters."
		return m, nil
	}
	if len(username) > 16 {
		m.errorMsg = "Username must be 16 characters or fewer."
		return m, nil
	}

	existing := m.root.keys.GetByPlayerName(username)
	if existing != nil {
		m.errorMsg = fmt.Sprintf("'%s' is already taken.", username)
		return m, nil
	}

	playerID := m.root.keys.NextPlayerID()
	if err := m.root.keys.Register(&db.KeyEntry{
		PlayerName:  username,
		PlayerID:    playerID,
		PublicKey:   m.publicKey,
		Fingerprint: m.fingerprint,
		CreatedAt:   time.Now().Format(time.RFC3339),
	}); err != nil {
		m.errorMsg = fmt.Sprintf("Error: %s", err.Error())
		return m, nil
	}

	return NewMenuModel(m.root, username), nil
}

func (m *RegisterModel) View() string {
	var b strings.Builder

	if m.hasKey {
		b.WriteString(TitleStyle.Render("New SSH Key Detected"))
		b.WriteString(fmt.Sprintf("\n%s %s\n\n", SubtleStyle.Render("Fingerprint:"), m.fingerprint))
		b.WriteString("This key is not yet registered.\n\n")
		b.WriteString(SubtleStyle.Render("Username: ") + m.input.View() + "\n\n")
		if m.errorMsg != "" {
			b.WriteString(ErrorStyle.Render(m.errorMsg) + "\n")
		}
		b.WriteString("\n" + SubtleStyle.Render("Enter to register  •  Esc to quit"))
	} else {
		b.WriteString(TitleStyle.Render("No SSH Key Found"))
		b.WriteString("\n\n")
		b.WriteString("To play, you need an SSH key.\n\n")
		b.WriteString(fmt.Sprintf("%s\n", HelpStyle.Render("Run this command in your terminal:")))
		b.WriteString(fmt.Sprintf("  %s\n\n", BoldStyle.Render("ssh-keygen -t ed25519")))
		b.WriteString(SubtleStyle.Render("Then reconnect. Your new key will be detected\nautomatically — just choose a username!"))
		b.WriteString("\n\n")
		b.WriteString(SubtleStyle.Render("Press any key to disconnect."))
	}

	return b.String()
}
