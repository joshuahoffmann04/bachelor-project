"""Web scraper for Modulhandbuch and Veranstaltungskalender."""

import time
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache_manager import CacheManager
from ..preprocessing.document import Document, DocumentChunk, SourceType, ContentType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebScraper:
    """Scrape web content with caching and rate limiting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize web scraper.

        Args:
            config: Scraping configuration
        """
        self.config = config or {}

        # Configuration
        self.user_agent = self.config.get('user_agent', 'RAGBot/1.0')
        self.rate_limit = self.config.get('rate_limit', 1)  # requests per second
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

        # Cache
        cache_config = self.config.get('cache', {})
        cache_dir = cache_config.get('path', 'data/scraped')
        cache_expiry = cache_config.get('expiry', 86400)
        self.cache = CacheManager(cache_dir, cache_expiry)

        # Session with retries
        self.session = self._create_session()

        # Rate limiting
        self.last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'de,en;q=0.9',
        })

        return session

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit."""
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def scrape_url(self, url: str, use_cache: bool = True) -> Optional[Document]:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            use_cache: Whether to use cached content

        Returns:
            Scraped Document or None on failure
        """
        logger.info("scraping_url", url=url)

        # Check cache first
        if use_cache:
            cached = self.cache.get(url)
            if cached:
                return self._cached_to_document(cached)

        # Rate limiting
        self._rate_limit_wait()

        try:
            # Fetch URL
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')

            # Extract content
            content, metadata = self._extract_content(soup, url)

            # Cache the result
            if use_cache:
                self.cache.set(url, content, metadata)

            # Create document
            doc = Document(
                doc_id=str(uuid.uuid4()),
                content=content,
                source_type=SourceType.WEB,
                source_url=url,
                title=metadata.get('title', url),
                scraped_at=datetime.utcnow(),
                metadata=metadata
            )

            logger.info("scraping_success", url=url, content_length=len(content))
            return doc

        except requests.RequestException as e:
            logger.error("scraping_failed", url=url, error=str(e))
            return None

    def _extract_content(self, soup: BeautifulSoup, url: str) -> tuple[str, Dict[str, Any]]:
        """Extract content from HTML.

        Args:
            soup: BeautifulSoup object
            url: Source URL

        Returns:
            Tuple of (content, metadata)
        """
        # Extract title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ''

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if not main_content:
            content = soup.get_text(separator='\n', strip=True)
        else:
            content = main_content.get_text(separator='\n', strip=True)

        # Extract structured elements
        headings = self._extract_headings(main_content or soup)
        tables = self._extract_tables(main_content or soup)
        lists = self._extract_lists(main_content or soup)

        metadata = {
            'title': title_text,
            'url': url,
            'headings': headings,
            'tables': tables,
            'lists': lists,
            'num_tables': len(tables),
            'num_headings': len(headings),
        }

        return content, metadata

    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract headings with hierarchy."""
        headings = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': int(tag.name[1]),
                'text': tag.get_text(strip=True)
            })
        return headings

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables as structured data."""
        tables = []
        for idx, table in enumerate(soup.find_all('table')):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)

            if rows:
                tables.append({
                    'index': idx,
                    'rows': rows,
                    'text': '\n'.join(' | '.join(row) for row in rows)
                })

        return tables

    def _extract_lists(self, soup: BeautifulSoup) -> List[str]:
        """Extract list items."""
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            items = [li.get_text(strip=True) for li in ul.find_all('li', recursive=False)]
            if items:
                lists.append('\n'.join(f"- {item}" for item in items))
        return lists

    def _cached_to_document(self, cached_data: Dict[str, Any]) -> Document:
        """Convert cached data to Document.

        Args:
            cached_data: Cached data dictionary

        Returns:
            Document object
        """
        metadata = cached_data.get('metadata', {})

        return Document(
            doc_id=str(uuid.uuid4()),
            content=cached_data['content'],
            source_type=SourceType.WEB,
            source_url=cached_data['url'],
            title=metadata.get('title', cached_data['url']),
            scraped_at=datetime.fromisoformat(cached_data['cached_at']),
            metadata=metadata
        )

    def extract_chunks(
        self,
        doc: Document,
        chunk_size: int = 512
    ) -> List[DocumentChunk]:
        """Extract chunks from scraped document.

        Args:
            doc: Scraped document
            chunk_size: Maximum chunk size

        Returns:
            List of document chunks
        """
        chunks = []

        # Get structured elements
        headings = doc.metadata.get('headings', [])
        tables = doc.metadata.get('tables', [])

        # Create chunks for tables (don't split)
        for table_info in tables:
            chunk = DocumentChunk(
                chunk_id=f"{doc.doc_id}_table_{table_info['index']}",
                content=table_info['text'],
                doc_id=doc.doc_id,
                source_type=SourceType.WEB,
                content_type=ContentType.TABLE,
                source_doc=doc.title,
                source_url=doc.source_url,
                scraped_at=doc.scraped_at,
                metadata={'table_index': table_info['index']}
            )
            chunks.append(chunk)

        # Create chunks for regular content
        # Split by headings if available
        if headings:
            # Split content by heading positions
            content_parts = self._split_by_headings(doc.content, headings)
            for idx, (section_title, section_content) in enumerate(content_parts):
                # Further chunk if too large
                if len(section_content) > chunk_size:
                    sub_chunks = self._chunk_text(section_content, chunk_size)
                    for sub_idx, sub_content in enumerate(sub_chunks):
                        chunk = DocumentChunk(
                            chunk_id=f"{doc.doc_id}_s{idx}_c{sub_idx}",
                            content=sub_content,
                            doc_id=doc.doc_id,
                            source_type=SourceType.WEB,
                            content_type=ContentType.PARAGRAPH,
                            section=section_title,
                            source_doc=doc.title,
                            source_url=doc.source_url,
                            scraped_at=doc.scraped_at
                        )
                        chunks.append(chunk)
                else:
                    chunk = DocumentChunk(
                        chunk_id=f"{doc.doc_id}_s{idx}",
                        content=section_content,
                        doc_id=doc.doc_id,
                        source_type=SourceType.WEB,
                        content_type=ContentType.PARAGRAPH,
                        section=section_title,
                        source_doc=doc.title,
                        source_url=doc.source_url,
                        scraped_at=doc.scraped_at
                    )
                    chunks.append(chunk)
        else:
            # Simple chunking
            text_chunks = self._chunk_text(doc.content, chunk_size)
            for idx, chunk_content in enumerate(text_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{doc.doc_id}_c{idx}",
                    content=chunk_content,
                    doc_id=doc.doc_id,
                    source_type=SourceType.WEB,
                    content_type=ContentType.PARAGRAPH,
                    source_doc=doc.title,
                    source_url=doc.source_url,
                    scraped_at=doc.scraped_at
                )
                chunks.append(chunk)

        logger.info("web_chunks_extracted", num_chunks=len(chunks), doc_id=doc.doc_id)
        return chunks

    def _split_by_headings(
        self,
        content: str,
        headings: List[Dict[str, str]]
    ) -> List[tuple[str, str]]:
        """Split content by heading positions.

        Args:
            content: Full content
            headings: List of headings with text

        Returns:
            List of (heading, content) tuples
        """
        sections = []
        lines = content.split('\n')

        current_heading = "Introduction"
        current_content = []

        for line in lines:
            # Check if line matches a heading
            is_heading = False
            for heading in headings:
                if heading['text'] in line:
                    # Save previous section
                    if current_content:
                        sections.append((current_heading, '\n'.join(current_content)))

                    current_heading = heading['text']
                    current_content = []
                    is_heading = True
                    break

            if not is_heading:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((current_heading, '\n'.join(current_content)))

        return sections

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks by paragraphs.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size

        Returns:
            List of text chunks
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
