package tui

import (
	"fmt"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/guyreading/container-rl-ssh/internal/client"
)

var (
	red    = lipgloss.Color("#FF4444")
	green  = lipgloss.Color("#44FF44")
	blue   = lipgloss.Color("#4488FF")
	yellow = lipgloss.Color("#FFD700")
	purple = lipgloss.Color("#FF44FF")
	cyan   = lipgloss.Color("#00AAAA")
	white  = lipgloss.Color("#FFFFFF")
	dim    = lipgloss.Color("#888888")

	colorStyles = []lipgloss.Style{
		lipgloss.NewStyle().Foreground(red),
		lipgloss.NewStyle().Foreground(green),
		lipgloss.NewStyle().Foreground(blue),
		lipgloss.NewStyle().Foreground(yellow),
		lipgloss.NewStyle().Foreground(purple),
	}
	colorNames = []string{"Red", "Green", "Blue", "Yellow", "Purple"}

	headerStyle = lipgloss.NewStyle().Bold(true).Background(lipgloss.Color("#0044AA")).Foreground(white).Padding(0, 1)
	supplyStyle = lipgloss.NewStyle().Border(lipgloss.NormalBorder(), false, false, false, false).BorderForeground(lipgloss.Color("#FFD700")).Padding(0, 1).Width(80)
	cardStyle   = lipgloss.NewStyle().Padding(0, 1)
	dimStyle    = lipgloss.NewStyle().Foreground(dim)
	errStyle    = lipgloss.NewStyle().Foreground(red)
	currentMark = lipgloss.NewStyle().Bold(true).Background(green).Foreground(white)
	actionStyle = lipgloss.NewStyle().Border(lipgloss.NormalBorder(), false, false, false, false).BorderForeground(cyan).Padding(0, 1)
	waitStyle   = lipgloss.NewStyle().Foreground(dim).Italic(true)
)

type GameState struct {
	Cash            []any `json:"cash"`
	Loans           []any `json:"loans"`
	FactoryColors   any   `json:"factory_colors"`
	WarehouseCount  []any `json:"warehouse_count"`
	FactoryStore    any   `json:"factory_store"`
	HarbourStore    any   `json:"harbour_store"`
	IslandStore     any   `json:"island_store"`
	ShipContents    any   `json:"ship_contents"`
	ShipLocation    any   `json:"ship_location"`
	ContainerSupply []any `json:"container_supply"`
	CurrentPlayer   any   `json:"current_player"`
	GameOver        any   `json:"game_over"`
	AuctionActive   any   `json:"auction_active"`
	AuctionSeller   any   `json:"auction_seller"`
	AuctionCargo    []any `json:"auction_cargo"`
	AuctionBids     any   `json:"auction_bids"`
	AuctionRound    any   `json:"auction_round"`
	ProduceActive   any   `json:"produce_active"`
	ShoppingActive  any   `json:"shopping_active"`
	ActionsTaken    any   `json:"actions_taken"`
	TurnPhase       any   `json:"turn_phase"`
	SecretColor     any   `json:"secret_value_color"`
}

func gsInt(v any, idx int) int {
	arr, ok := v.([]any)
	if !ok || idx >= len(arr) {
		return 0
	}
	switch val := arr[idx].(type) {
	case float64:
		return int(val)
	}
	return 0
}

func gsInt2(v any, i, j int) int {
	arr, ok := v.([]any)
	if !ok || i >= len(arr) {
		return 0
	}
	row, ok := arr[i].([]any)
	if !ok || j >= len(row) {
		return 0
	}
	switch val := row[j].(type) {
	case float64:
		return int(val)
	}
	return 0
}

func gsInt3(v any, i, j, k int) int {
	arr, ok := v.([]any)
	if !ok || i >= len(arr) {
		return 0
	}
	row, ok := arr[i].([]any)
	if !ok || j >= len(row) {
		return 0
	}
	col, ok := row[j].([]any)
	if !ok || k >= len(col) {
		return 0
	}
	switch val := col[k].(type) {
	case float64:
		return int(val)
	}
	return 0
}

const (
	ActionPass           = 7
	ActionBuyFactory     = 0
	ActionBuyWarehouse   = 1
	ActionProduce        = 2
	ActionBuyFromFactory = 3
	ActionMoveLoad       = 4
	ActionMoveSea        = 5
	ActionMoveAuction    = 6
	ActionTakeLoan       = 8
	ActionRepayLoan      = 9

	MAX_COLORS            = 5
	PRICE_SLOTS           = 10
	SHIP_CAPACITY         = 5
	LEAVE_IDLE            = 4
	LOCATION_OPEN_SEA     = 0
	LOCATION_AUCTION_ISLE = 1
	LOCATION_HARBOUR      = 2
	HARBOUR_PRICE_MIN     = 1
	HARBOUR_PRICE_CHOICES = 5
	MAX_FACTORIES         = 4
)

type PlayModel struct {
	root     *RootModel
	state    *SharedState
	gs       *GameState
	myTurn   bool
	errorMsg string
	feedback string
	quitting bool
	gameOver bool

	// submenu state
	submenu   string
	subCursor int
	subData   any
}

func NewPlayModel(root *RootModel, state *SharedState) *PlayModel {
	return &PlayModel{
		root:  root,
		state: state,
	}
}

func (m *PlayModel) Init() tea.Cmd {
	debugLog("PLAY_INIT\n")
	return tea.Batch(
		m.listenServer(),
		func() tea.Msg {
			if m.state.TCPClient == nil {
				return errMsg{err: "not connected"}
			}
			m.state.TCPClient.Send("get_state", map[string]any{})
			return nil
		},
	)
}

func (m *PlayModel) listenServer() tea.Cmd {
	return func() tea.Msg {
		debugLog("PLAY_LISTEN START\n")
		if m.state.TCPClient == nil {
			return errMsg{err: "not connected"}
		}
		for msg := range m.state.TCPClient.Recv() {
			debugLog("PLAY_MSG type=%s\n", msg.Type)
			switch msg.Type {
			case "state_update":
				var payload struct {
					StateData      GameState `json:"state_data"`
					CurrentPlayer  int       `json:"current_player"`
					ActionsTaken   int       `json:"actions_taken"`
					AuctionActive  int       `json:"auction_active"`
					ProduceActive  int       `json:"produce_active"`
					ShoppingActive int       `json:"shopping_active"`
					GameOver       int       `json:"game_over"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return stateUpdateMsg{
					gameState:      &payload.StateData,
					currentPlayer:  payload.CurrentPlayer,
					actionsTaken:   payload.ActionsTaken,
					auctionActive:  payload.AuctionActive,
					produceActive:  payload.ProduceActive,
					shoppingActive: payload.ShoppingActive,
					gameOver:       payload.GameOver,
				}
			case "your_turn":
				var payload struct{ PlayerIndex int `json:"player_index"` }
				jsonUnmarshal(msg.Payload, &payload)
				return yourTurnMsg{playerIndex: payload.PlayerIndex}
			case "action_result":
				var payload struct {
					TurnEnded bool   `json:"turn_ended"`
					Desc      string `json:"desc"`
					Reward    float64 `json:"reward"`
					GameOver  bool   `json:"game_over"`
				}
				jsonUnmarshal(msg.Payload, &payload)
				return actionResultMsg{desc: payload.Desc, gameOver: payload.GameOver}
			case "error":
				var payload struct{ Message string `json:"message"` }
				jsonUnmarshal(msg.Payload, &payload)
				return errMsg{err: payload.Message}
			}
		}
		return errMsg{err: "disconnected"}
	}
}

func (m *PlayModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case stateUpdateMsg:
		debugLog("PLAY_UPDATE stateUpdate cur=%d myTurn=%v\n", msg.currentPlayer, msg.currentPlayer == m.state.PlayerIdx)
		m.gs = msg.gameState
		m.myTurn = msg.currentPlayer == m.state.PlayerIdx
		if msg.gameOver > 0 {
			m.gameOver = true
		}
		m.errorMsg = ""
		if m.gs.AuctionsActive() > 0 || m.gs.Producing() > 0 || m.gs.Shopping() > 0 {
			return m, nil
		}
		return m, m.listenServer()
	case yourTurnMsg:
		m.myTurn = msg.playerIndex == m.state.PlayerIdx
	case actionResultMsg:
		m.feedback = msg.desc
		if msg.gameOver {
			m.gameOver = true
		}
	case errMsg:
		m.errorMsg = msg.err

	case tea.KeyMsg:
		if m.quitting {
			return m, nil
		}
		switch msg.String() {
		case "ctrl+c":
			m.quitting = true
		case "q", "esc":
			if !m.myTurn || m.submenu == "" {
				m.quitting = true
			}
		}

		if m.myTurn && m.submenu == "" && !m.gameOver {
			switch msg.String() {
			case "1":
				return m, m.sendAction(ActionBuyFactory, 0, 0, 0, 0)
			case "2":
				return m, m.sendAction(ActionBuyWarehouse, 0, 0, 0, 0)
			case "3":
				return m, m.sendAction(ActionProduce, 0, 0, 0, 0)
			case "4":
				return m, m.sendAction(ActionBuyFromFactory, 0, 0, 0, 0)
			case "5":
				return m, m.sendAction(ActionMoveLoad, 0, 0, 0, 0)
			case "6":
				return m, m.sendAction(ActionMoveSea, 0, 0, 0, 0)
			case "7":
				return m, m.sendAction(ActionMoveAuction, 0, 0, 0, 0)
			case "8":
				return m, m.sendAction(ActionTakeLoan, 0, 0, 0, 0)
			case "9":
				return m, m.sendAction(ActionRepayLoan, 0, 0, 0, 0)
			case "0", " ":
				return m, m.sendAction(ActionPass, 0, 0, 0, 0)
			}
		}
	}

	if m.quitting {
		if m.state.TCPClient != nil {
			m.state.TCPClient.Close()
		}
		return m, tea.Quit
	}
	return m, nil
}

func (m *PlayModel) sendAction(atype, opponent, color, slot, purchase int) tea.Cmd {
	action := []int{atype, opponent, color, slot, purchase}
	return func() tea.Msg {
		if m.state.TCPClient == nil {
			return errMsg{err: "not connected"}
		}
		m.state.TCPClient.Send("action_multi", map[string]any{"action": action})
		return nil
	}
}

func (m *PlayModel) View() string {
	if m.quitting {
		if m.state.TCPClient != nil {
			m.state.TCPClient.Close()
		}
		return TitleStyle.Render("Goodbye!") + "\n\nThanks for playing Container!"
	}

	if m.gs == nil {
		return TitleStyle.Render(fmt.Sprintf("Game %s", m.state.GameCode)) +
			"\n\n" + dimStyle.Render("Loading game state...")
	}

	return m.renderBoard()
}

func (gs *GameState) NumPlayers() int { return len(gs.Cash) }
func (gs *GameState) CurPlayer() int  { return gsInt(gs.CurrentPlayer, 0) }
func (gs *GameState) AuctionsActive() int { return gsInt(gs.AuctionActive, 0) }
func (gs *GameState) Producing() int     { return gsInt(gs.ProduceActive, 0) }
func (gs *GameState) Shopping() int      { return gsInt(gs.ShoppingActive, 0) }

func (m *PlayModel) renderBoard() string {
	gs := m.gs
	np := gs.NumPlayers()
	nc := m.state.NumColors

	var b strings.Builder

	// Header
	curName := fmt.Sprintf("Player %d", gs.CurPlayer()+1)
	actionN := gsInt(gs.ActionsTaken, 0) + 1
	header := fmt.Sprintf("CONTAINER  |  %s's turn  |  Action %d/2", curName, actionN)
	if gs.AuctionsActive() > 0 {
		header += "  |  AUCTION"
	}
	if gs.Producing() > 0 {
		header += "  |  PRODUCING"
	}
	if gs.Shopping() > 0 {
		header += "  |  SHOPPING"
	}
	if m.myTurn {
		header += "  (YOU)"
	}
	b.WriteString(headerStyle.Render(header) + "\n")

	// Supply
	b.WriteString(m.renderSupply(nc) + "\n")

	// Player cards side by side
	cards := make([]string, np)
	for p := 0; p < np; p++ {
		cards[p] = m.renderPlayerCard(p, nc)
	}
	b.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, cards...))
	b.WriteString("\n")

	// Feedback
	if m.feedback != "" {
		b.WriteString(dimStyle.Render(m.feedback) + "\n")
	}
	if m.errorMsg != "" {
		b.WriteString(errStyle.Render(m.errorMsg) + "\n")
	}

	// Actions
	b.WriteString("\n")
	b.WriteString(m.renderActions())
	return b.String()
}

func (m *PlayModel) renderSupply(nc int) string {
	gs := m.gs
	var parts []string
	exhausted := 0
	for c := 0; c < nc; c++ {
		cnt := gsInt(gs.ContainerSupply, c)
		style := colorStyles[c]
		bar := ""
		for i := 0; i < 10 && i < cnt; i++ {
			bar += "█"
		}
		bar += " " + strconv.Itoa(cnt)
		parts = append(parts, style.Render(colorNames[c]+": "+bar))
		if cnt <= 0 {
			exhausted++
		}
	}
	parts = append(parts, fmt.Sprintf("Exhausted: %d/2", exhausted))
	return supplyStyle.Render(strings.Join(parts, "  │  "))
}

func (m *PlayModel) renderPlayerCard(player, nc int) string {
	gs := m.gs
	var b strings.Builder

	name := fmt.Sprintf("Player %d", player+1)
	cash := gsInt(gs.Cash, player)
	loans := gsInt(gs.Loans, player)
	wh := gsInt(gs.WarehouseCount, player)

	// Title
	title := fmt.Sprintf("%s  $%d", name, cash)
	if player == m.state.PlayerIdx {
		title = currentMark.Render("◄") + " " + title
	}
	b.WriteString(title + "\n")

	// Separator
	b.WriteString(strings.Repeat("─", 28) + "\n")

	// Stats
	b.WriteString(fmt.Sprintf("  💵 $%-4d 🏦 %d loans  🏭 %d wh\n", cash, loans, wh))

	// Secret
	sc := gsInt(gs.SecretColor, player)
	b.WriteString(fmt.Sprintf("  🤫 %s\n", colorStyles[sc].Render(colorNames[sc])))

	// Factories
	factories := m.renderFactories(player, nc)
	b.WriteString("  Factories: " + factories + "\n")

	// Factory Store
	b.WriteString("  Factory Store:\n")
	fs := m.renderStore(gs.FactoryStore, player, nc)
	if fs == "" {
		b.WriteString("    (empty)\n")
	} else {
		b.WriteString(fs)
	}

	// Harbour Store
	b.WriteString("  Harbour Store:\n")
	hs := m.renderStore(gs.HarbourStore, player, nc)
	if hs == "" {
		b.WriteString("    (empty)\n")
	} else {
		b.WriteString(hs)
	}

	// Island
	island := m.renderIsland(player, nc)
	b.WriteString("  " + island + "\n")

	// Ship
	ship := m.renderShip(player, nc)
	b.WriteString("  🚢 " + ship + "\n")

	return cardStyle.Render(b.String())
}

func (m *PlayModel) renderFactories(player, nc int) string {
	gs := m.gs
	var parts []string
	for c := 0; c < nc; c++ {
		if gsInt2(gs.FactoryColors, player, c) > 0 {
			parts = append(parts, colorStyles[c].Render(colorNames[c]))
		}
	}
	if len(parts) == 0 {
		return "(none)"
	}
	return strings.Join(parts, ", ")
}

func (m *PlayModel) renderStore(store any, player, nc int) string {
	var b strings.Builder
	for c := 0; c < nc; c++ {
		var items []string
		for s := 0; s < PRICE_SLOTS; s++ {
			cnt := gsInt3(store, player, c, s)
			if cnt > 0 {
				items = append(items, fmt.Sprintf("%dx$%d", cnt, s+1))
			}
		}
		if len(items) > 0 {
			b.WriteString(fmt.Sprintf("    %s: %s\n",
				colorStyles[c].Render(colorNames[c]),
				strings.Join(items, ", ")))
		}
	}
	return b.String()
}

func (m *PlayModel) renderIsland(player, nc int) string {
	gs := m.gs
	var parts []string
	for c := 0; c < nc; c++ {
		cnt := gsInt2(gs.IslandStore, player, c)
		if cnt > 0 {
			parts = append(parts, colorStyles[c].Render(fmt.Sprintf("%dx %s", cnt, colorNames[c])))
		}
	}
	if len(parts) == 0 {
		return "🏝️ (empty)"
	}
	return "🏝️ " + strings.Join(parts, " ")
}

func (m *PlayModel) renderShip(player, nc int) string {
	gs := m.gs
	// Cargo
	var cargo []string
	for i := 0; i < SHIP_CAPACITY; i++ {
		c := gsInt2(gs.ShipContents, player, i)
		if c == 0 {
			cargo = append(cargo, "·")
		} else {
			cargo = append(cargo, colorStyles[c-1].Render("■"))
		}
	}
	cargoStr := strings.Join(cargo, " ")

	// Location
	loc := gsInt(gs.ShipLocation, player)
	locStr := "Unknown"
	switch {
	case loc == LOCATION_OPEN_SEA:
		locStr = "Open Sea"
	case loc == LOCATION_AUCTION_ISLE:
		locStr = "Auction Isl."
	case loc >= LOCATION_HARBOUR:
		locStr = fmt.Sprintf("P%d Harbour", loc-1)
	}

	return fmt.Sprintf("%s  @ %s", cargoStr, locStr)
}

func (m *PlayModel) renderActions() string {
	var b strings.Builder

	if !m.myTurn {
		b.WriteString(waitStyle.Render(fmt.Sprintf("Waiting for Player %d to play...\n", m.gs.CurPlayer()+1)))
	} else if m.gameOver {
		b.WriteString("Game Over!\n")
	} else if m.gs.AuctionsActive() > 0 {
		b.WriteString("Auction in progress...\n")
	} else if m.gs.Producing() > 0 || m.gs.Shopping() > 0 {
		b.WriteString("Waiting for other player...\n")
	} else {
		b.WriteString("[1] BuyFactory  [2] BuyWarehouse  [3] Produce  [4] BuyFromFactory  [5] LoadShip\n")
		b.WriteString("[6] MoveToSea  [7] Auction  [0/Space] Pass  [8] TakeLoan  [9] RepayLoan\n")
		b.WriteString(dimStyle.Render("\nq to leave"))
	}

	return actionStyle.Render(b.String())
}

var _ = client.GameClient{}
var _ = lipgloss.NewStyle()
