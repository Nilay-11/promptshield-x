"""
PromptShield X - GitHub Extractor
Collects the attacker-writable text of a repository through the public GitHub REST API:
README, issues + comments, pull requests + review comments, commit messages, workflow files and code comments.
Accepts a repo URL, an issue URL (/issues/N) or a pull-request URL (/pull/N). Request count is bounded.
Set GITHUB_TOKEN to raise the API rate limit (optional).
"""

import base64
import os
import re
from typing import Optional

import httpx

from app.modules.extractors import BaseExtractor, ExtractedSegment, ExtractionResult, SourceType

CODE_EXTS = (".py", ".js", ".ts", ".java", ".go", ".rb", ".c", ".cpp", ".h", ".cs", ".rs", ".php", ".sh", ".yml", ".yaml")
COMMENT_RE = re.compile(r"(?:#|//)\s?(.+)$|/\*([\s\S]*?)\*/|\"\"\"([\s\S]*?)\"\"\"|'''([\s\S]*?)'''", re.M)
MAX_ITEMS = 10      # issues / PRs / commits fetched per list
MAX_CODE_FILES = 8  # source files scanned for comments


class GitHubExtractor(BaseExtractor):
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.api_base = "https://api.github.com"

    def _seg(self, result, content, location, hidden=False):
        content = (content or "").strip()
        if content:
            result.segments.append(ExtractedSegment(content=content, source_type=SourceType.CODE,
                                                    location=location, is_hidden=hidden, threat_indicators=[]))

    def _get(self, client, path, result, **params):
        r = client.get(f"{self.api_base}{path}", params=params or None)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            result.extraction_warnings.append("GitHub API rate limit reached (set GITHUB_TOKEN); partial scan.")
        return None

    def _thread(self, client, result, base, kind, number):
        """One issue or PR: body, comments and (for PRs) review comments."""
        item = self._get(client, f"{base}/issues/{number}", result)
        if not item:
            return
        self._seg(result, f"{item.get('title', '')}\n{item.get('body') or ''}", {"element": kind, "number": number})
        for c in self._get(client, f"{base}/issues/{number}/comments", result, per_page=30) or []:
            self._seg(result, c.get("body"), {"element": f"{kind}_comment", "number": number, "author": c.get("user", {}).get("login")})
        if kind == "pull_request":
            for c in self._get(client, f"{base}/pulls/{number}/comments", result, per_page=30) or []:
                self._seg(result, c.get("body"), {"element": "review_comment", "number": number, "file": c.get("path")})

    def _file_text(self, client, result, base, path):
        f = self._get(client, f"{base}/contents/{path}", result)
        if isinstance(f, dict) and f.get("content"):
            return base64.b64decode(f["content"]).decode("utf-8", errors="ignore")
        return None

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        url = data.decode("utf-8").strip()
        result = ExtractionResult(source_type=SourceType.CODE, filename=url)
        m = re.search(r"github\.com/([^/\s]+)/([^/\s#?]+)(?:/(issues|pull)/(\d+))?", url)
        if not m:
            result.extraction_warnings.append("Invalid GitHub URL format.")
            return result
        owner, repo, kind, number = m.groups()
        repo = repo.removesuffix(".git")
        base = f"/repos/{owner}/{repo}"
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        try:
            with httpx.Client(headers=headers, timeout=10) as client:
                if kind:  # a single issue / PR URL: scan just that thread
                    self._thread(client, result, base, "pull_request" if kind == "pull" else "issue", int(number))
                else:
                    readme = self._get(client, f"{base}/readme", result)
                    if readme:
                        self._seg(result, base64.b64decode(readme["content"]).decode("utf-8", errors="ignore"),
                                  {"file": readme.get("path", "README.md")})
                    # issues and PRs (the issues endpoint returns both)
                    for it in self._get(client, f"{base}/issues", result, state="all", per_page=MAX_ITEMS) or []:
                        self._thread(client, result, base, "pull_request" if "pull_request" in it else "issue", it["number"])
                    for c in self._get(client, f"{base}/commits", result, per_page=MAX_ITEMS) or []:
                        self._seg(result, c.get("commit", {}).get("message"), {"element": "commit", "sha": c.get("sha", "")[:7]})
                    # workflows + code comments from the default-branch tree
                    info = self._get(client, base, result) or {}
                    tree = self._get(client, f"{base}/git/trees/{info.get('default_branch', 'HEAD')}", result, recursive=1) or {}
                    paths = [t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"]
                    for p in [p for p in paths if p.startswith(".github/workflows/")][:5]:
                        self._seg(result, self._file_text(client, result, base, p), {"file": p, "element": "workflow"})
                    for p in [p for p in paths if p.endswith(CODE_EXTS) and not p.startswith(".github/")][:MAX_CODE_FILES]:
                        text = self._file_text(client, result, base, p) or ""
                        comments = [next(g for g in mm.groups() if g) for mm in COMMENT_RE.finditer(text) if any(mm.groups())]
                        comments = [c.strip() for c in comments if len(c.split()) >= 4]
                        if comments:
                            self._seg(result, "\n".join(comments), {"file": p, "element": "code_comments"}, hidden=True)
        except Exception as e:
            result.extraction_warnings.append(f"GitHub API extraction error: {str(e)}")

        result.raw_character_count = sum(len(s.content) for s in result.segments)
        return result
