import httpx
import ipaddress
import urllib.parse
import socket
from typing import Optional
from app.modules.extractors import BaseExtractor, ExtractionResult, SourceType, ThreatCategory
from app.modules.extractors.web_extractor import WebExtractor

class URLExtractor(BaseExtractor):
    def __init__(self, max_response_size: int = 5 * 1024 * 1024, timeout_seconds: int = 5):
        self.max_response_size = max_response_size
        self.timeout = timeout_seconds
        self.web_parser = WebExtractor()

    def _is_ip_safe(self, ip_str: str) -> bool:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
            # Block Cloud metadata endpoints (usually link-local 169.254.169.254, covered by is_link_local, but let's be explicit)
            if str(ip) == "169.254.169.254":
                return False
            return True
        except ValueError:
            return False

    def _resolve_and_check_safety(self, hostname: str) -> bool:
        """Resolve hostname to IP and check if it violates SSRF rules."""
        try:
            # Simplistic resolution - if we resolve multiple, check all.
            # socket.gethostbyname only returns one, but that's sufficient for basic protection.
            # In a highly robust environment we'd use getaddrinfo and hook the socket directly.
            ip = socket.gethostbyname(hostname)
            return self._is_ip_safe(ip)
        except Exception:
            return False

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        # For URL extractor, `data` contains the URL string (encoded as bytes)
        url = data.decode("utf-8").strip()
        result = ExtractionResult(source_type=SourceType.WEB, filename=url)

        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                result.extraction_warnings.append("Invalid scheme. Only http/https are allowed.")
                return result

            if not parsed.hostname:
                result.extraction_warnings.append("Invalid URL structure (no hostname).")
                return result

            # SSRF Protection: Resolve and check IP
            if not self._resolve_and_check_safety(parsed.hostname):
                result.extraction_warnings.append("SSRF Protection activated: Hostname resolves to private/internal IP.")
                return result

            # Safe HTTP fetch with max sizes and redirects limited
            with httpx.Client(timeout=self.timeout, follow_redirects=True, max_redirects=3) as client:
                response = client.get(url)
                response.raise_for_status()
                
                # Check headers for content length to avoid reading infinite streams
                if int(response.headers.get("Content-Length", 0)) > self.max_response_size:
                    result.extraction_warnings.append(f"Content too large. Max size {self.max_response_size} bytes.")
                    return result

                body = response.content
                if len(body) > self.max_response_size:
                    result.extraction_warnings.append(f"Content streaming too large. Terminated.")
                    return result
                
                # Pass off to WebExtractor for DOM logic
                web_result = self.web_parser.extract(data=body, filename=url)
                return web_result

        except httpx.HTTPError as he:
            result.extraction_warnings.append(f"HTTP fetching error: {str(he)}")
        except Exception as e:
            result.extraction_warnings.append(f"URL extraction error: {str(e)}")

        return result
