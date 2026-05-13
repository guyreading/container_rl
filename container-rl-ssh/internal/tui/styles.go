package tui

import "github.com/charmbracelet/lipgloss"

var (
	TitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#7B2D8E")).
			MarginBottom(1)

	ErrorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FF4444"))

	SuccessStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#44FF44"))

	SubtleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#888888"))

	HelpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFD700"))

	PlayerColor = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00AAFF"))

	HighlightStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(lipgloss.Color("#7B2D8E"))

	BoldStyle = lipgloss.NewStyle().Bold(true)

	CursorStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FFD700"))
)
