---
title: "XOR and RC4: From Principles to Go Practice (with Safer Alternatives)"
date: 2025-12-16T18:12:00+08:00
description: "An intro to XOR and RC4 for Go/backend engineers, with runnable examples and safer alternatives."
tags: ["go", "security", "crypto", "rc4", "xor"]
categories: ["go", "security"]
reading_time: "8 min"
seo_keywords: "XOR, RC4, stream cipher, Go, encryption, Base64"
meta_description: "Explain XOR and RC4 with runnable Go examples and why RC4 is no longer secure."
draft: false
---

# XOR and RC4: From Principles to Go Practice (with Safer Alternatives)

## Subtitle / Abstract
Use minimal math to explain XOR and RC4, provide runnable Go examples, and clarify why RC4 is considered insecure with recommended alternatives.

## Target readers
- Backend engineers reading legacy RC4 code
- Beginners who confuse encoding and encryption
- Intermediate developers building a stream-cipher mental model

## Background / Motivation
Many systems still contain RC4 or custom decryption logic. Common mistakes include treating Base64 as encryption and ignoring integrity checks. Understanding XOR and RC4 helps you evaluate security correctly and avoid copying outdated designs into new systems.

## Core concepts
- XOR: bitwise operation, reversible
- Stream cipher: XOR a pseudorandom keystream with plaintext bytes
- RC4: classic stream cipher, no longer recommended
- Base64: encoding, not encryption
- Integrity: encryption alone does not prevent tampering

## Practical steps
1) Receive a Base64 string (often RC4 output)
2) Decode Base64 to raw bytes
3) Initialize RC4 with a shared key
4) XOR keystream with bytes
5) Convert output to UTF-8 text if it is textual

## Runnable example (Go)
```go
package main

import (
	"crypto/rc4"
	"encoding/base64"
	"fmt"
)

func rc4XOR(key string, data []byte) ([]byte, error) {
	c, err := rc4.NewCipher([]byte(key))
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(data))
	c.XORKeyStream(out, data)
	return out, nil
}

func encryptToBase64RC4(key, plaintext string) (string, error) {
	out, err := rc4XOR(key, []byte(plaintext))
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(out), nil
}

func decryptBase64RC4(key, encoded string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", err
	}
	out, err := rc4XOR(key, raw)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func main() {
	key := "demo-key"
	plaintext := "hello rc4"

	enc, _ := encryptToBase64RC4(key, plaintext)
	dec, _ := decryptBase64RC4(key, enc)

	fmt.Println(enc)
	fmt.Println(dec)
}
```

Run:
- `go run rc4_demo.go`

## Explanation
XOR is reversible because `a XOR b XOR b = a`. RC4 generates a pseudorandom keystream and XORs it with data byte by byte. Since encryption and decryption use the same keystream, keystream reuse or bias can leak information.

## Common pitfalls
- Base64 is encoding, not encryption
- RC4 has known biases and is deprecated
- Encryption alone does not provide integrity; use MAC or AEAD
- Reusing keys can reveal plaintext

## Best practices
- Use AES-GCM or ChaCha20-Poly1305 for new systems
- Migrate legacy RC4 systems as soon as possible
- Consider confidentiality and integrity together

## Conclusion
XOR is the core operation behind stream ciphers. RC4 is easy to understand but unsafe; it is suitable for reading legacy code, not new design. Modern systems should use AEAD algorithms instead.

## References
- https://www.rfc-editor.org/rfc/rfc6229
- https://www.rfc-editor.org/rfc/rfc7465
- https://en.wikipedia.org/wiki/RC4
- https://pkg.go.dev/crypto/rc4

## Meta
- Reading time: 8 minutes
- Tags: go, security, crypto, rc4, xor
- SEO keywords: XOR, RC4, stream cipher, Go, encryption, Base64
- Meta description: Explain XOR and RC4 with runnable Go examples and why RC4 is no longer secure.

## Call to Action (CTA)
After running the demo, replace RC4 with AES-GCM and document the differences and migration cost for your team.
