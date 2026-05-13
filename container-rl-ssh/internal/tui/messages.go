package tui

import (
	"encoding/json"
	"time"
)

var waitTimeout = 15 * time.Second

type errMsg struct{ err string }

type gameCreatedMsg struct {
	gameID      int64
	code        string
	playerIndex int
}

type gameJoinedMsg struct {
	gameID      int64
	code        string
	playerIndex int
	numPlayers  int
	numColors   int
	status      string
}

type lobbyUpdateMsg struct {
	players        []PlayerInfo
	numNeeded      int
	code           string
}

type PlayerInfo struct {
	PlayerIndex int    `json:"player_index"`
	PlayerID    int64  `json:"player_id"`
	Name        string `json:"name"`
}

type gameListMsg struct {
	games []GameInfo
}

type GameInfo struct {
	ID          int64  `json:"id"`
	Code        string `json:"code"`
	Status      string `json:"status"`
	NumPlayers  int    `json:"num_players"`
	NumColors   int    `json:"num_colors"`
	SlotsFilled int    `json:"slots_filled"`
}

type gameStartedMsg struct{}

type stateUpdateMsg struct {
	gameState      *GameState
	currentPlayer  int
	actionsTaken   int
	auctionActive  int
	produceActive  int
	shoppingActive int
	gameOver       int
}

type yourTurnMsg struct {
	playerIndex int
}

type actionResultMsg struct {
	turnEnded bool
	desc      string
	reward    float64
	gameOver  bool
}

func jsonUnmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}
