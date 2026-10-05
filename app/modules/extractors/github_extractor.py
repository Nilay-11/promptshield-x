import base64
import httpx
import re
from typing import Optional
from app.modules.extractors import BaseExtractor, ExtractedSegment, ExtractionResult, SourceType, ThreatCategory

class GitHubExtractor(BaseExtractor):
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.api_base = "https://api.github.com"
        
    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        # data contains the github url, e.g. https://github.com/owner/repo
        url = data.decode("utf-8").strip()
        result = ExtractionResult(source_type=SourceType.CODE, filename=url)
        
        # Parse owner and repo
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if not match:
            result.extraction_warnings.append("Invalid GitHub URL format.")
            return result
            
        owner, repo = match.groups()
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
            
        try:
            with httpx.Client(headers=headers, timeout=10) as client:
                # 1. Fetch README
                readme_resp = client.get(f"{self.api_base}/repos/{owner}/{repo}/readme")
                if readme_resp.status_code == 200:
                    readme_data = readme_resp.json()
                    content = base64.b64decode(readme_data["content"]).decode("utf-8", errors="ignore")
                    result.segments.append(
                        ExtractedSegment(
                            content=content,
                            source_type=SourceType.CODE,
                            location={"file": readme_data.get("path", "README.md")},
                            is_hidden=False,
                            threat_indicators=[],
                        )
                    )
                
                # 2. Fetch specific issues or other things if they were in the URL
                # For basic implementation, we just fetch README as the primary vector.
                # A full implementation would traverse the dir tree recursively and extract comments via AST.
                
                result.raw_character_count = sum(len(s.content) for s in result.segments)

        except Exception as e:
            result.extraction_warnings.append(f"GitHub API extraction error: {str(e)}")

        return result
