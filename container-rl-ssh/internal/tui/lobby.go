package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

type LobbyModel struct {
	root      *RootModel
	state     *SharedState
	players   []PlayerInfo
	numNeeded int
	code      string
	errorMsg  string
	quitting  bool
	joined    bool
}

func NewLobbyModel(root *RootModel, state *SharedState) *LobbyModel {
	debugLog("NEW_LOBBY code=%s client=%v\n", state.GameCode, state.TCPClient != nil)
	return &LobbyModel{
		root:  root,
		state: state,
	}
}

func (m *LobbyModel) Init() tea.Cmd {
	return tea.Batch(
		m.listenServer(),
		tea.Tick(time.Second, func(t time.Time) tea.Msg {
			return lobbyTickMsg(t)
		}),
	)
}

func (m *LobbyModel) listenServer() tea.Cmd {
	return func() tea.Msg {
		debugLog("LOBBY_LISTEN START client=%v\n", m.state.TCPClient != nil)
		if m.state.TCPClient == nil {
			return errMsg{err: "not connected"}
		}
		for msg := range m.state.TCPClient.Recv() {
			debugLog("LOBBY_MSG type=%s\n", msg.Type)
			switch msg.Type {
			case "game_created":
				var payload struct {
					GameID      int    `json:"game_id"`
					Code        string `json:"code"`
					PlayerIndex int    `json:"player_index"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return gameCreatedMsg{
					gameID:      int64(payload.GameID),
					code:        payload.Code,
					playerIndex: payload.PlayerIndex,
				}
			case "game_joined":
				var payload struct {
					GameID      int    `json:"game_id"`
					Code        string `json:"code"`
					PlayerIndex int    `json:"player_index"`
					NumPlayers  int    `json:"num_players"`
					NumColors   int    `json:"num_colors"`
					Status      string `json:"status"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return gameJoinedMsg{
					gameID:      int64(payload.GameID),
					code:        payload.Code,
					playerIndex: payload.PlayerIndex,
					numPlayers:  payload.NumPlayers,
					numColors:   payload.NumColors,
					status:      payload.Status,
				}
			case "lobby_update":
				var payload struct {
					Players    []PlayerInfo `json:"players"`
					NumPlayers int          `json:"num_players_needed"`
					Code       string       `json:"code"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return lobbyUpdateMsg{
					players:   payload.Players,
					numNeeded: payload.NumPlayers,
					code:      payload.Code,
				}
			case "game_started":
				return gameStartedMsg{}
			case "error":
				var payload struct {
					Message string `json:"message"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return errMsg{err: payload.Message}
			}
		}
		return errMsg{err: "connection closed"}
	}
}

func (m *LobbyModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case lobbyTickMsg:
		if m.quitting {
			return m, tea.Quit
		}
		return m, tea.Tick(time.Second, func(t time.Time) tea.Msg {
			return lobbyTickMsg(t)
		})
	case lobbyUpdateMsg:
		m.players = msg.players
		m.numNeeded = msg.numNeeded
		m.code = msg.code
		return m, m.listenServer()
	case gameCreatedMsg:
		m.code = msg.code
		m.state.GameID = msg.gameID
		m.state.PlayerIdx = msg.playerIndex
		return m, m.listenServer()
	case gameJoinedMsg:
		m.code = msg.code
		m.state.GameID = msg.gameID
		m.state.PlayerIdx = msg.playerIndex
		m.state.NumPlayers = msg.numPlayers
		m.state.NumColors = msg.numColors
		m.joined = true
		return m, m.listenServer()
	case gameStartedMsg:
		debugLog("LOBBY_GAME_STARTED transition to PlayModel\n")
		pm := NewPlayModel(m.root, m.state)
		return pm, pm.Init()
	case errMsg:
		m.errorMsg = msg.err
		m.quitting = true
	case tea.KeyMsg:
		if !m.quitting {
			switch msg.String() {
			case "q", "esc", "ctrl+c":
				m.quitting = true
			}
		}
	}
	if m.quitting {
		if m.state.TCPClient != nil {
			m.state.TCPClient.Close()
			m.state.TCPClient = nil
		}
		return m, tea.Quit
	}
	return m, nil
}

func (m *LobbyModel) View() string {
	var b strings.Builder

	if m.quitting {
		b.WriteString(TitleStyle.Render("Leaving lobby..."))
		b.WriteString("\n\n" + SubtleStyle.Render("Goodbye!"))
		return b.String()
	}

	b.WriteString(TitleStyle.Render(fmt.Sprintf("Game %s", m.code)))
	b.WriteString("\n\n")
	b.WriteString(fmt.Sprintf("%s %d / %d\n\n", SubtleStyle.Render("Players:"), len(m.players), m.numNeeded))

	for _, p := range m.players {
		marker := ""
		if p.Name == m.state.PlayerName {
			marker = " (you)"
		}
		b.WriteString(fmt.Sprintf("  %s%s\n", p.Name, SubtleStyle.Render(marker)))
	}

	needed := m.numNeeded - len(m.players)
	if needed < 0 {
		needed = 0
	}
	b.WriteString(fmt.Sprintf("\n%s %d more player(s) needed.\n", SubtleStyle.Render("Waiting:"), needed))

	if m.errorMsg != "" {
		b.WriteString("\n" + ErrorStyle.Render(m.errorMsg))
	}
	b.WriteString("\n\n" + SubtleStyle.Render("q to leave"))
	return b.String()
}

type lobbyTickMsg time.Time
