---
title: "用 Python 还原 RC4 + JWT + 自定义 SSO Token 加解密（含可运行示例）"
date: 2025-11-19
draft: false
tags: ["Python", "加密", "JWT", "SSO", "RC4", "安全", "WebSocket"]
keywords: ["RC4", "JWT", "Python 加密", "Base64", "Hex", "SSO Token"]
description: "从核心概念到可运行代码，演示用 Python 实现 RC4 加/解密、JWT 与自定义 SSO Token，并讨论风险与替代方案。示例中密钥与发行方均为占位值。"
---

# 用 Python 还原 RC4 + JWT + 自定义 SSO Token 加解密（含可运行示例）

**副标题 / 摘要**

这篇文章带你从 0 拆解 RC4 流加密、Base64/Hex 编码，以及基于 JWT 与自定义 SSO 的鉴权设计，并给出可以复制运行的 Python 示例。示例中的密钥与发行方均为占位值，切勿用于生产。

---

## 目标读者

- Python 后端/测试工程师（中级）
- 对鉴权、令牌与基础加密流程感兴趣的开发者
- 想理解 RC4 工作方式与替代方案的安全入门读者

---

## 背景 / 动机

在实际项目中，我们常需要为 HTTP 或 WebSocket 请求附带令牌进行身份校验。常见做法包括使用 JWT（对称/非对称签名）或自定义的 SSO Token（例如对某段明文进行对称加密后再以 Hex/Base64 编码）。本文整理并复现一种组合方案：RC4+Base64/Hex 与 JWT/SSO 的加解密与校验流程，帮助你在测试或 PoC 中快速上手，同时理解其安全取舍。

---

## 核心概念

- RC4：经典流加密（已不再安全）。通过密钥调度（KSA）与伪随机序列（PRGA）生成密钥流，与明文字节按位异或得到密文。解密过程与加密一致（同一函数）。
- Base64 与 Hex：两种将二进制数据编码为可传输文本的方式。Base64 更紧凑；Hex 可读、调试直观。
- JWT：JSON Web Token。Header.Payload.Signature。常包含 iss（发行方）、aud（受众/自定义）、iat/exp（签发/过期）。
- 自定义 SSO Token：一种自定义明文格式（示例采用 issuer_expire_ts_userSeqId_userId），经对称加密后再编码为 Hex，便于在 HTTP 头中传输。
- 限制与风险：RC4 已过时且不建议用于生产；如需兼容遗留系统，应仅在测试/过渡场景，且搭配 TLS、短周期、签名与回放防护。

---

## 实践指南 / 步骤

1) 安装依赖

```bash
pip install pyjwt
```

2) 设定占位常量（不要使用真实密钥/发行方）

```python
ISSUER = "demo-issuer"
SECRET = "demo-secret-change-me"
RC4KEY = "demo-rc4-key-change-me"
UTE_ISSUER = "ute-demo"
```

3) 实现 RC4 与常用编码包装

- encrypt_string/decrypt_string：RC4 后 Base64
- encrypt_hex_string/decrypt_hex_string：RC4 后 Hex

4) 构造与校验 JWT（x-auth-token）

- aud = [Base64(RC4(user_id)), Base64(RC4(user_seq_id))]
- iss/iat/exp 等标准字段

5) 构造与校验 SSO Token（x-sso-token）

- 明文 `UTE_ISSUER_expire_ts_userSeqId_userId` → RC4 → Hex
- 校验发行方与过期时间

6) 运行演示，观察生成与校验结果

---

## 可运行示例（完整代码）

> 仅用于学习与测试，切勿将 RC4 用于生产环境。请优先使用现代 AEAD（AES‑GCM/ChaCha20‑Poly1305）。

```python
import base64
import time
from typing import Optional, Dict, Tuple, Union
import jwt

# 占位常量（不要用真实值）
ISSUER = "demo-issuer"
SECRET = "demo-secret-change-me"
RC4KEY = "demo-rc4-key-change-me"
UTE_ISSUER = "ute-demo"

def rc4(key: str, data: bytes) -> bytes:
    # KSA
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]
    # PRGA
    i = j = 0
    out = []
    for ch in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        k = S[(S[i] + S[j]) % 256]
        out.append(ch ^ k)
    return bytes(out)

def encrypt_string(plain: str) -> str:
    c = rc4(RC4KEY, plain.encode("utf-8"))
    return base64.b64encode(c).decode("utf-8")

def decrypt_string(enc_b64: str) -> Optional[str]:
    try:
        c = base64.b64decode(enc_b64)
        p = rc4(RC4KEY, c)
        return p.decode("utf-8")
    except Exception:
        return None

def encrypt_hex_string(plain: str) -> str:
    c = rc4(RC4KEY, plain.encode("utf-8"))
    return c.hex()

def decrypt_hex_string(enc_hex: str) -> Optional[str]:
    try:
        c = bytes.fromhex(enc_hex)
        p = rc4(RC4KEY, c)
        return p.decode("utf-8")
    except Exception:
        return None

def make_auth_token(user_id: str, user_seq_id: str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": [encrypt_string(user_id), encrypt_string(user_seq_id)],
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def verify_auth_token(token: str) -> Union[Dict[str, str], Tuple[None, str]]:
    try:
        if not token or len(token) < 64:
            return None, "Token 无效（空或太短）"
        decoded = jwt.decode(
            token,
            SECRET,
            algorithms=["HS256"],
            issuer=ISSUER,
            options={"verify_aud": False},
        )
        audience = decoded.get("aud")
        if not audience or len(audience) < 2:
            return None, "Token 缺少 audience"
        user_id = decrypt_string(audience[0])
        user_seq_id = decrypt_string(audience[1])
        if not user_id or not user_seq_id:
            return None, "解密后的用户信息为空"
        return {"type": "x-auth", "user_id": user_id, "user_seq_id": user_seq_id}
    except jwt.ExpiredSignatureError:
        return None, "Token 已过期"
    except Exception as e:
        return None, f"验证失败: {e}"

def make_sso_token(user_id: str, user_seq_id: str, ttl_seconds: int = 3600) -> str:
    expire = int(time.time()) + ttl_seconds
    plain = f"{UTE_ISSUER}_{expire}_{user_seq_id}_{user_id}"
    return encrypt_hex_string(plain)

def verify_sso_token(token: str) -> Union[Dict[str, str], Tuple[None, str]]:
    try:
        if not token:
            return None, "[SSO] Token为空"
        plain = decrypt_hex_string(token)
        if not plain:
            return None, "[SSO] Token解密失败"
        parts = plain.split("_")
        if len(parts) < 4:
            return None, f"[SSO] Token分段不足4部分: {parts}"
        if parts[0] != UTE_ISSUER:
            return None, f"[SSO] 发行者不匹配: 期望={UTE_ISSUER}, 实际={parts[0]}"
        expire_time = int(parts[1])
        if expire_time < int(time.time()):
            return None, f"[SSO] Token已过期: {expire_time}"
        user_seq_id = parts[2]
        user_id = parts[3]
        return {"type": "x-sso", "user_id": user_id, "user_seq_id": user_seq_id}
    except Exception as e:
        return None, f"[SSO] 验证异常: {e}"

if __name__ == "__main__":
    uid = "user_123"
    useq = "seq_456"

    print("=== 1) 纯加解密演示 ===")
    enc_b64 = encrypt_string(uid)
    print("Base64密文:", enc_b64)
    print("Base64解密:", decrypt_string(enc_b64))
    enc_hex = encrypt_hex_string(uid)
    print("Hex密文:", enc_hex)
    print("Hex解密:", decrypt_hex_string(enc_hex))

    print("\n=== 2) JWT 演示 (x-auth-token) ===")
    jwt_token = make_auth_token(uid, useq, ttl_seconds=10)
    print("JWT:", jwt_token)
    print("JWT 校验:", verify_auth_token(jwt_token))

    print("\n=== 3) SSO 演示 (x-sso-token) ===")
    sso_token = make_sso_token(uid, useq, ttl_seconds=10)
    print("SSO Token:", sso_token)
    print("SSO 校验:", verify_sso_token(sso_token))
```

---

## 解释与原理

- RC4 工作流：
  - KSA 用密钥打乱状态数组 S；PRGA 基于 S 生成伪随机字节流，与明文按字节 XOR 得到密文；解密与加密同一过程（XOR 的逆仍是 XOR）。
- Base64/Hex 的取舍：
  - Base64 更紧凑，适合缩短传输体积；Hex 更直观，便于调试、肉眼对比。
- JWT 的校验点：
  - 验证签名、发行方（iss）、时间（iat/exp）。本文示例将受众信息（aud）存放经 RC4+Base64 的 user_id 与 user_seq_id，校验时再解密取值。
- 自定义 SSO Token 的设计：
  - 明文本身包含 issuer 与过期时间戳，加密后作为 Hex 放入请求头；服务端解密、验证 issuer/expire/user 信息。

 > 延伸阅读：现代 AEAD 方案与最佳实践（AES‑GCM/ChaCha20‑Poly1305）见《[现代加密替代方案：AES‑GCM 与 ChaCha20‑Poly1305 实战指南](/posts/python/python-modern-aead/)》。

---

## 常见问题与注意事项

- RC4 为什么不安全？
  - 关键流偏差、密钥复用风险、对明文结构的敏感性等问题。建议使用 AES‑GCM 或 ChaCha20‑Poly1305。
- Base64/Hex 有安全性差异吗？
  - 它们只是编码方式，不提供安全性；机密性来自加密或签名。
- Token 太短/太长会怎样？
  - 过短可能是伪造/截断；过长可能因承载过多信息影响传输；应精简且仅携带必要信息。
- 时间偏差导致过期？
  - 生产应统一时钟（NTP），并考虑小幅度时钟偏移容错（leeway）。
- 如何做 Key 轮换？
  - 引入 kid（Key ID）与多活密钥，逐步切换；对称密钥定期轮换、限制可见范围。

---

## 最佳实践与建议

- 避免 RC4：选用 AES‑GCM 或 ChaCha20‑Poly1305。
- 若使用 JWT：
  - 对称密钥仅限服务端；高敏场景优先非对称（RS256/ES256）。
  - 强制校验 iss/aud/exp，设置短 TTL，使用 TLS，启用回放防护（nonce/一次性 token）。
- 令牌设计：
  - 减少可识别敏感信息（避免在明文字段直接放用户ID）。
  - 加签或使用标准化格式（如 JWS/JWE）。
- 测试/调试：
  - 使用占位常量，避免泄露真实密钥与发行方。
  - 在自动化测试中用工厂方法生成 token，集中管理。

---

## 小结 / 结论

本文用 Python 复现了 RC4+Base64/Hex、JWT 与自定义 SSO Token 的加解密与校验流程，给出可运行示例并说明了 RC4 的风险与替代方案。若是生产场景，建议优先使用现代 AEAD，并完善令牌生命周期与密钥管理。

---

## 参考与延伸阅读

- RFC 7519: JSON Web Token (JWT)
- PyJWT 文档: https://pyjwt.readthedocs.io/
- RC4（维基百科，安全讨论与历史）
 - 延伸：《[现代加密替代方案：AES‑GCM 与 ChaCha20‑Poly1305 实战指南](/posts/python/python-modern-aead/)》

---

## 元信息

- 预计阅读时长：10 分钟
- 标签：Python、加密、JWT、SSO、RC4、安全
- SEO 关键词：RC4、JWT、Python 加密、Base64、Hex
- 元描述：用 Python 还原 RC4 与 JWT/SSO Token 的加解密流程，含完整示例与安全建议，示例中的密钥与发行方均为占位值。

---

## 行动号召（CTA）

- 试着运行示例，替换占位密钥与发行方，观察校验结果
- 将生成函数接入你的测试夹具（fixtures），自动构造 x-sso-token / x-auth-token
- 阅读延伸文章并在生产中采用现代 AEAD 方案
