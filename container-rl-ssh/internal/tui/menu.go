package tui

import (
	"fmt"
	"os"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/guyreading/container-rl-ssh/internal/client"
)

var debugFile *os.File

func init() {
	var err error
	debugFile, err = os.OpenFile("/tmp/container-rl-debug.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		debugFile = os.Stderr
	}
}

func debugLog(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	debugFile.WriteString(msg)
	os.Stderr.WriteString(msg)
}

type MenuModel struct {
	root       *RootModel
	playerName string
	cursor     int
	choices    []string
}

func NewMenuModel(root *RootModel, playerName string) *MenuModel {
	return &MenuModel{
		root:       root,
		playerName: playerName,
		cursor:     0,
		choices:    []string{"Create Game", "Join / Continue Game", "Quit"},
	}
}

func (m *MenuModel) Init() tea.Cmd { return nil }

func (m *MenuModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		debugLog("MENU_KEY type=%d string=%q runes=%v\n", msg.Type, msg.String(), msg.Runes)
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q", "esc":
			return m, tea.Quit
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.choices)-1 {
				m.cursor++
			}
		case "enter", " ":
			debugLog("MENU_ENTER cursor=%d choice=%s\n", m.cursor, m.choices[m.cursor])
			switch m.cursor {
			case 0:
				return NewCreateModel(m.root, m.playerName), nil
			case 1:
				jm := NewJoinModel(m.root, m.playerName)
				return jm, jm.fetchGames()
			case 2:
				return m, tea.Quit
			}
		}
	}
	return m, nil
}

func (m *MenuModel) View() string {
	var b strings.Builder
	b.WriteString(TitleStyle.Render(fmt.Sprintf("Welcome, %s!", m.playerName)))
	b.WriteString("\n\n")

	for i, choice := range m.choices {
		if m.cursor == i {
			b.WriteString(CursorStyle.Render("▸ " + choice))
		} else {
			b.WriteString("  " + choice)
		}
		b.WriteString("\n")
	}

	b.WriteString("\n" + SubtleStyle.Render("jk/↑↓ to navigate  •  Enter to select  •  Ctrl+C to quit"))
	return b.String()
}

type CreateModel struct {
	root       *RootModel
	playerName string
	numPlayers int
	errorMsg   string
	loading    bool
}

func NewCreateModel(root *RootModel, playerName string) *CreateModel {
	return &CreateModel{
		root:       root,
		playerName: playerName,
		numPlayers: 2,
	}
}

func (m *CreateModel) Init() tea.Cmd { return nil }

func (m *CreateModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q", "esc":
			return NewMenuModel(m.root, m.playerName), nil
		case "enter":
			if m.loading {
				return m, nil
			}
			return m.handleSubmit()
		case "left", "h":
			if m.loading {
				return m, nil
			}
			if m.numPlayers > 2 {
				m.numPlayers--
			}
		case "right", "l":
			if m.loading {
				return m, nil
			}
			if m.numPlayers < 5 {
				m.numPlayers++
			}
		}
	}
	return m, nil
}

func (m *CreateModel) handleSubmit() (tea.Model, tea.Cmd) {
	m.loading = true
	m.errorMsg = "Connecting..."

	gc, err := client.Connect(m.root.gameAddr)
	if err != nil {
		m.errorMsg = fmt.Sprintf("Cannot connect to game server: %s", err)
		m.loading = false
		return m, nil
	}

	if err := gc.Send("create_game", map[string]any{
		"player_name": m.playerName,
		"num_players": m.numPlayers,
		"num_colors":  5,
	}); err != nil {
		m.errorMsg = fmt.Sprintf("Error creating game: %s", err)
		m.loading = false
		gc.Close()
		return m, nil
	}

	state := &SharedState{
		PlayerName: m.playerName,
		TCPClient:  gc,
		NumPlayers: m.numPlayers,
		NumColors:  5,
	}

	lm := NewLobbyModel(m.root, state)
	return lm, lm.Init()
}

func (m *CreateModel) View() string {
	var b strings.Builder
	b.WriteString(TitleStyle.Render("Create Game"))
	b.WriteString(fmt.Sprintf("\n%s %s\n\n", SubtleStyle.Render("Player:"), m.playerName))

	bar := make([]rune, 5)
	for i := 0; i < 5; i++ {
		if i < m.numPlayers {
			bar[i] = '█'
		} else {
			bar[i] = '░'
		}
	}
	left := "◂"
	right := "▸"
	if m.numPlayers <= 2 {
		left = " "
	}
	if m.numPlayers >= 5 {
		right = " "
	}

	b.WriteString(fmt.Sprintf("  %s  %s %d  %s\n\n",
		SubtleStyle.Render(left),
		CursorStyle.Render(string(bar)),
		m.numPlayers,
		SubtleStyle.Render(right),
	))
	b.WriteString(SubtleStyle.Render("  ←→/hl adjust  •  Enter to create  •  Esc to back"))

	b.WriteString("\n")
	if m.errorMsg != "" {
		b.WriteString("\n" + ErrorStyle.Render(m.errorMsg) + "\n")
	}
	if m.loading {
		b.WriteString(SubtleStyle.Render("Creating game..."))
	}
	return b.String()
}

type JoinModel struct {
	root       *RootModel
	playerName string
	cursor     int
	games      []GameInfo
	loading    bool
	errorMsg   string
}

func NewJoinModel(root *RootModel, playerName string) *JoinModel {
	debugLog("NEW_JOIN_MODEL player=%s\n", playerName)
	return &JoinModel{
		root:       root,
		playerName: playerName,
		cursor:     0,
		loading:    true,
	}
}

func (m *JoinModel) Init() tea.Cmd {
	return nil
}

func (m *JoinModel) fetchGames() tea.Cmd {
	return func() tea.Msg {
		debugLog("FETCH_GAMES START player=%s addr=%s\n", m.playerName, m.root.gameAddr)
		gc, err := client.Connect(m.root.gameAddr)
		if err != nil {
			debugLog("FETCH_GAMES CONNECT_FAILED: %s\n", err)
			return errMsg{err: fmt.Sprintf("Cannot connect: %s", err)}
		}
		defer gc.Close()

		debugLog("FETCH_GAMES CONNECTED\n")
		if err := gc.Send("list_games", map[string]any{
			"player_name": m.playerName,
		}); err != nil {
			debugLog("FETCH_GAMES SEND_FAILED: %s\n", err)
			return errMsg{err: fmt.Sprintf("Error: %s", err)}
		}

		debugLog("FETCH_GAMES WAITING\n")
		msg, ok := <-gc.Recv()
		if !ok {
			debugLog("FETCH_GAMES CHANNEL_CLOSED\n")
			return errMsg{err: "connection closed"}
		}
		debugLog("FETCH_GAMES RECEIVED type=%s\n", msg.Type)
		if msg.Type == "error" {
			var payload struct {
				Message string `json:"message"`
			}
			jsonUnmarshal(msg.Payload, &payload)
			return errMsg{err: payload.Message}
		}
		if msg.Type == "game_list" {
			var payload struct {
				Games []GameInfo `json:"games"`
			}
			jsonUnmarshal(msg.Payload, &payload)
			debugLog("FETCH_GAMES GOT %d games\n", len(payload.Games))
			return gameListMsg{games: payload.Games}
		}
		return errMsg{err: fmt.Sprintf("unexpected: %s", msg.Type)}
	}
}

func (m *JoinModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case gameListMsg:
		debugLog("JOIN_UPDATE gameListMsg count=%d\n", len(msg.games))
		m.games = msg.games
		m.loading = false
		return m, nil
	case errMsg:
		debugLog("JOIN_UPDATE errMsg=%s\n", msg.err)
		m.errorMsg = msg.err
		m.loading = false
		return m, nil
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q", "esc":
			return NewMenuModel(m.root, m.playerName), nil
		}
		if m.loading {
			return m, nil
		}
		switch msg.String() {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.games) {
				m.cursor++
			}
		case "enter", " ":
			debugLog("JOIN_ENTER cursor=%d numGames=%d\n", m.cursor, len(m.games))
			if m.loading {
				debugLog("JOIN_ENTER IGNORED (loading)\n")
				return m, nil
			}
			if m.cursor == len(m.games) {
				return NewMenuModel(m.root, m.playerName), nil
			}
			if m.cursor >= 0 && m.cursor < len(m.games) {
				return m.joinGame(m.games[m.cursor])
			}
			return m, nil
		}
	}
	return m, nil
}

func (m *JoinModel) joinGame(game GameInfo) (tea.Model, tea.Cmd) {
	debugLog("JOIN_GAME code=%s player=%s\n", game.Code, m.playerName)
	m.loading = true

	gc, err := client.Connect(m.root.gameAddr)
	if err != nil {
		debugLog("JOIN_GAME CONNECT_FAILED: %s\n", err)
		m.errorMsg = fmt.Sprintf("Cannot connect: %s", err)
		m.loading = false
		return m, nil
	}

	if err := gc.Send("join_game", map[string]any{
		"player_name": m.playerName,
		"code":        game.Code,
	}); err != nil {
		debugLog("JOIN_GAME SEND_FAILED: %s\n", err)
		m.errorMsg = fmt.Sprintf("Error: %s", err)
		m.loading = false
		gc.Close()
		return m, nil
	}

	state := &SharedState{
		PlayerName: m.playerName,
		TCPClient:  gc,
		GameCode:   game.Code,
	}

	lm := NewLobbyModel(m.root, state)
	return lm, lm.Init()
}

func (m *JoinModel) View() string {
	var b strings.Builder
	b.WriteString(TitleStyle.Render("Join / Continue Game"))
	b.WriteString(fmt.Sprintf("\n%s %s\n\n", SubtleStyle.Render("Player:"), m.playerName))

	if m.loading && len(m.games) == 0 {
		b.WriteString(SubtleStyle.Render("Loading games..."))
		b.WriteString("\n")
		return b.String()
	}

	if len(m.games) == 0 {
		b.WriteString(SubtleStyle.Render("No games available.\n\n"))
		b.WriteString(SubtleStyle.Render("Ask a friend to create one, or create your own!"))
		b.WriteString("\n\n")
		b.WriteString(SubtleStyle.Render("esc to go back"))
		return b.String()
	}

	for i, g := range m.games {
		status := ""
		if g.Status == "active" {
			status = SubtleStyle.Render(" [rejoin]")
		} else {
			status = SubtleStyle.Render(fmt.Sprintf(" [%d/%d players]", g.SlotsFilled, g.NumPlayers))
		}
		line := fmt.Sprintf("  %s%s", g.Code, status)

		if m.cursor == i {
			b.WriteString(CursorStyle.Render("▸ " + line))
		} else {
			b.WriteString("  " + line)
		}
		b.WriteString("\n")
	}

	backCursor := "  "
	if m.cursor == len(m.games) {
		backCursor = "▸ "
		b.WriteString(CursorStyle.Render(backCursor + "Back"))
	} else {
		b.WriteString(backCursor + "Back")
	}
	b.WriteString("\n")

	b.WriteString("\n")
	if m.errorMsg != "" {
		b.WriteString(ErrorStyle.Render(m.errorMsg) + "\n")
	}
	b.WriteString(SubtleStyle.Render("jk/↑↓ to navigate  •  Enter to select  •  Esc to back"))
	return b.String()
}
