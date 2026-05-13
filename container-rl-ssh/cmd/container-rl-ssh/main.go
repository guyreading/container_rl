package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/charmbracelet/log"

	"github.com/guyreading/container-rl-ssh/internal/server"
)

func main() {
	listenAddr := flag.String("addr", ":2222", "SSH listen address")
	hostKeyDir := flag.String("host-key-dir", "./ssh_host_keys", "Host key directory")
	keysPath := flag.String("keys", "ssh_keys.json", "SSH keys JSON file")
	gameAddr := flag.String("game-addr", "127.0.0.1:9876", "Python game server address")
	flag.Parse()

	log.SetLevel(log.InfoLevel)
	log.SetPrefix("container-ssh")

	fmt.Fprintf(os.Stderr, "Container RL SSH Server\n")
	fmt.Fprintf(os.Stderr, "  SSH: %s\n", *listenAddr)
	fmt.Fprintf(os.Stderr, "  Game: %s\n", *gameAddr)
	fmt.Fprintf(os.Stderr, "  Keys: %s\n", *keysPath)

	if err := server.Run(server.Config{
		ListenAddr: *listenAddr,
		HostKeyDir: *hostKeyDir,
		KeysPath:   *keysPath,
		GameAddr:   *gameAddr,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
