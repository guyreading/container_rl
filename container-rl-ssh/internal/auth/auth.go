package auth

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/charmbracelet/log"
	gossh "golang.org/x/crypto/ssh"

	"github.com/charmbracelet/ssh"
	"github.com/guyreading/container-rl-ssh/internal/db"
)

type AuthStatus string

const (
	StatusKnown    AuthStatus = "known"
	StatusNewKey   AuthStatus = "new_key"
	StatusAnonymous AuthStatus = "anonymous"
)

const (
	CtxAuthStatus  = "auth_status"
	CtxPlayerName  = "player_name"
	CtxPlayerID    = "player_id"
	CtxPublicKey   = "public_key"
	CtxFingerprint = "fingerprint"
)

func Fingerprint(key gossh.PublicKey) string {
	h := sha256.Sum256(key.Marshal())
	encoded := base64.RawStdEncoding.EncodeToString(h[:])
	return fmt.Sprintf("SHA256:%s", encoded)
}

func ParsePublicKey(encoded string) (gossh.PublicKey, error) {
	key, _, _, _, err := gossh.ParseAuthorizedKey([]byte(strings.TrimSpace(encoded)))
	if err != nil {
		return nil, fmt.Errorf("parse public key: %w", err)
	}
	return key, nil
}

type Handlers struct {
	Keys *db.KeyStore
}

func (h *Handlers) PublicKeyHandler(ctx ssh.Context, key ssh.PublicKey) bool {
	encoded := strings.TrimSpace(string(gossh.MarshalAuthorizedKey(key)))
	fingerprint := Fingerprint(key)

	log.Info("public_key_auth", "fingerprint", fingerprint)
	log.Info("public_key_auth", "key_type", key.Type())

	entry := h.Keys.Lookup(encoded)
	if entry != nil {
		log.Info("public_key_auth result", "status", "known", "name", entry.PlayerName)
		ctx.SetValue(CtxAuthStatus, StatusKnown)
		ctx.SetValue(CtxPlayerName, entry.PlayerName)
		ctx.SetValue(CtxPlayerID, entry.PlayerID)
		return true
	}

	log.Info("public_key_auth result", "status", "new_key")
	ctx.SetValue(CtxAuthStatus, StatusNewKey)
	ctx.SetValue(CtxPublicKey, encoded)
	ctx.SetValue(CtxFingerprint, fingerprint)
	return true
}

func (h *Handlers) PasswordHandler(ctx ssh.Context, password string) bool {
	log.Info("password_auth", "user", ctx.User())
	ctx.SetValue(CtxAuthStatus, StatusAnonymous)
	return true
}
