package db

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

type KeyEntry struct {
	PlayerName  string `json:"player_name"`
	PlayerID    int64  `json:"player_id"`
	PublicKey   string `json:"public_key"`
	Fingerprint string `json:"fingerprint"`
	CreatedAt   string `json:"created_at"`
}

type KeyStore struct {
	mu   sync.RWMutex
	path string
	keys map[string]*KeyEntry
}

func NewKeyStore(path string) (*KeyStore, error) {
	ks := &KeyStore{
		path: path,
		keys: make(map[string]*KeyEntry),
	}
	if _, err := os.Stat(path); err == nil {
		if err := ks.load(); err != nil {
			return nil, fmt.Errorf("load key store: %w", err)
		}
	}
	return ks, nil
}

func (ks *KeyStore) load() error {
	data, err := os.ReadFile(ks.path)
	if err != nil {
		return err
	}
	var entries []*KeyEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("parse key store: %w", err)
	}
	for _, e := range entries {
		ks.keys[e.PublicKey] = e
	}
	return nil
}

func (ks *KeyStore) save() error {
	var entries []*KeyEntry
	for _, e := range ks.keys {
		entries = append(entries, e)
	}
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(ks.path, data, 0600)
}

func (ks *KeyStore) Lookup(publicKey string) *KeyEntry {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	return ks.keys[publicKey]
}

func (ks *KeyStore) Register(entry *KeyEntry) error {
	ks.mu.Lock()
	defer ks.mu.Unlock()
	if _, exists := ks.keys[entry.PublicKey]; exists {
		return fmt.Errorf("key already registered")
	}
	ks.keys[entry.PublicKey] = entry
	return ks.save()
}

func (ks *KeyStore) NextPlayerID() int64 {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	var maxID int64
	for _, e := range ks.keys {
		if e.PlayerID > maxID {
			maxID = e.PlayerID
		}
	}
	return maxID + 1
}

func (ks *KeyStore) GetByPlayerName(name string) *KeyEntry {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	for _, e := range ks.keys {
		if e.PlayerName == name {
			return e
		}
	}
	return nil
}
