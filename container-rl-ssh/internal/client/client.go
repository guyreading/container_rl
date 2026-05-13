package client

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/guyreading/container-rl-ssh/internal/protocol"
)

type GameClient struct {
	conn     net.Conn
	mu       sync.Mutex
	incoming chan *protocol.Message
	done     chan struct{}
	closed   bool
}

func Connect(addr string) (*GameClient, error) {
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return nil, fmt.Errorf("connect to %s: %w", addr, err)
	}
	c := &GameClient{
		conn:     conn,
		incoming: make(chan *protocol.Message, 64),
		done:     make(chan struct{}),
	}
	go c.readLoop()
	return c, nil
}

func (c *GameClient) readLoop() {
	defer close(c.incoming)
	for {
		msg, err := protocol.ReadMessage(c.conn)
		if err != nil {
			return
		}
		select {
		case c.incoming <- msg:
		case <-c.done:
			return
		}
	}
}

func (c *GameClient) Send(msgType string, payload any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("connection closed")
	}
	data, err := protocol.PackMessage(msgType, payload)
	if err != nil {
		return err
	}
	_, err = c.conn.Write(data)
	return err
}

func (c *GameClient) Recv() <-chan *protocol.Message {
	return c.incoming
}

func (c *GameClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.closed = true
	close(c.done)
	c.conn.Close()
}

func (c *GameClient) WaitFor(msgType string, timeout time.Duration) (json.RawMessage, error) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case msg, ok := <-c.incoming:
			if !ok {
				return nil, fmt.Errorf("connection closed")
			}
			if msg.Type == msgType {
				return msg.Payload, nil
			}
			if msg.Type == "error" {
				var payload struct {
					Message string `json:"message"`
				}
				json.Unmarshal(msg.Payload, &payload)
				return nil, fmt.Errorf("server error: %s", payload.Message)
			}
		case <-timer.C:
			return nil, fmt.Errorf("timeout waiting for %s", msgType)
		}
	}
}
