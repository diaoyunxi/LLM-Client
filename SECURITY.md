# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.

### How to Report

1. **Private vulnerability reporting**: Use GitHub's built-in private vulnerability reporting feature
2. **Email**: Contact the repository owner via GitHub profile

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Status update**: Within 7 days
- **Resolution**: As soon as possible, depending on severity

## Security Architecture

LLM-Client implements multiple layers of security for tool execution:

### Shell Command Execution (shell_exec)

- **Command whitelist**: Only pre-approved commands can be executed
- **shell=False**: All subprocess calls use argument lists, preventing shell injection
- **Working directory whitelist**: Prevents path traversal via `working_dir` parameter
- **Output size limits**: Prevents LLM context overflow (default 10KB)
- **Dangerous command blacklist**: Blocks destructive commands (rm -rf /, mkfs, etc.)
- **Permission-restricted commands**: `chmod`/`chown` removed from whitelist

### Calculator Tool

- **Operand limits**: `pow()` and `exp()` have maximum operand values to prevent CPU exhaustion attacks
- **AST-based evaluation**: Only allows safe mathematical operations, no arbitrary code execution

### Web Search Tool

- **SSRF protection**: Validates URLs to reject private/internal IP addresses
- **Scheme whitelist**: Only allows `http://` and `https://` protocols

### File Operations

- **Path traversal protection**: Uses `os.path.realpath()` to resolve symlinks
- **Working directory constraints**: File operations restricted to allowed directories

## Deployment Security

When deploying LLM-Client:

1. **Run in isolated environment** — use containers or VMs for untrusted workloads
2. **Restrict network access** — limit outbound connections to required endpoints only
3. **Use environment variables** for API keys and model endpoints
4. **Keep dependencies updated** — run `pip-audit -r requirements.txt` regularly
5. **Review tool whitelist** — customize `ALLOWED_COMMANDS` for your use case
6. **Monitor shell_exec logs** — watch for suspicious command patterns

## Dependency Security

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

---

Thank you for helping keep LLM-Client secure!
