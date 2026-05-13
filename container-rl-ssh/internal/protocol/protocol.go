package protocol

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const MaxMessageSize = 10 * 1024 * 1024

type Message struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

func PackMessage(msgType string, payload any) ([]byte, error) {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	msg := Message{
		Type:    msgType,
		Payload: payloadBytes,
	}
	body, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("marshal message: %w", err)
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(body)))
	return append(header, body...), nil
}

func ReadMessage(r io.Reader) (*Message, error) {
	header := make([]byte, 4)
	if _, err := io.ReadFull(r, header); err != nil {
		if err == io.EOF {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("read header: %w", err)
	}
	length := binary.BigEndian.Uint32(header)
	if length > MaxMessageSize {
		return nil, fmt.Errorf("frame too large: %d bytes", length)
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	var msg Message
	if err := json.Unmarshal(body, &msg); err != nil {
		return nil, fmt.Errorf("unmarshal message: %w", err)
	}
	return &msg, nil
}

func UnmarshalPayload[T any](msg *Message) (T, error) {
	var payload T
	if err := json.Unmarshal(msg.Payload, &payload); err != nil {
		return payload, fmt.Errorf("unmarshal payload: %w", err)
	}
	return payload, nil
}
