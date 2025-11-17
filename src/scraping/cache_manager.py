"""Cache manager for web scraping."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages caching of scraped web content."""

    def __init__(self, cache_dir: str = "data/scraped", expiry_seconds: int = 86400):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            expiry_seconds: Cache expiry time in seconds (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_seconds = expiry_seconds

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL.

        Args:
            url: URL to hash

        Returns:
            Cache key (hashed URL)
        """
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content for URL.

        Args:
            url: URL to retrieve

        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug("cache_miss", url=url, reason="not_found")
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check expiry
            cached_at = datetime.fromisoformat(data['cached_at'])
            age = datetime.utcnow() - cached_at

            if age.total_seconds() > self.expiry_seconds:
                logger.debug("cache_miss", url=url, reason="expired", age_seconds=age.total_seconds())
                return None

            logger.debug("cache_hit", url=url, age_seconds=age.total_seconds())
            return data

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("cache_read_error", url=url, error=str(e))
            return None

    def set(self, url: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Cache content for URL.

        Args:
            url: URL being cached
            content: Content to cache
            metadata: Optional metadata
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        data = {
            'url': url,
            'content': content,
            'metadata': metadata or {},
            'cached_at': datetime.utcnow().isoformat(),
            'cache_key': cache_key
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("cache_set", url=url)

        except Exception as e:
            logger.error("cache_write_error", url=url, error=str(e))

    def clear_expired(self) -> int:
        """Clear expired cache entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cached_at = datetime.fromisoformat(data['cached_at'])
                age = datetime.utcnow() - cached_at

                if age.total_seconds() > self.expiry_seconds:
                    cache_file.unlink()
                    cleared += 1
                    logger.debug("cache_cleared", file=cache_file.name)

            except Exception as e:
                logger.warning("cache_clear_error", file=cache_file.name, error=str(e))

        logger.info("cache_cleanup", cleared=cleared)
        return cleared

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.warning("cache_delete_error", file=cache_file.name, error=str(e))

        logger.info("cache_cleared_all", cleared=cleared)
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = 0
        expired = 0
        total_size = 0

        for cache_file in self.cache_dir.glob("*.json"):
            total += 1
            total_size += cache_file.stat().st_size

            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cached_at = datetime.fromisoformat(data['cached_at'])
                age = datetime.utcnow() - cached_at

                if age.total_seconds() > self.expiry_seconds:
                    expired += 1

            except Exception:
                pass

        return {
            'total_entries': total,
            'expired_entries': expired,
            'active_entries': total - expired,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }
