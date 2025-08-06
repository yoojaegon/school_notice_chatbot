import os
import re
import time
import random
import requests
import logging
import logging.handlers
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv

# --- 로깅 설정 (변경 없음) ---
def setup_logging():
    """로그 설정: 콘솔과 파일에 다른 레벨로 로그를 남깁니다."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "crawler.log")
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --- 유틸리티 함수 (변경 없음) ---
BASE_SAVE_DIR = "crawled_data"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def parse_announcement_urls(url_string):
    board_dict = {}
    if not url_string:
        return board_dict
    lines = url_string.strip().split(',')
    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            name, url = line.split('|')
            board_dict[name.strip()] = url.strip()
        except ValueError:
            logging.warning(f"잘못된 형식의 URL 라인을 건너뜁니다: {line}")
    return board_dict

def download_file(url, save_path):
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"  [성공] 파일 다운로드: {os.path.basename(save_path)}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"  [오류] 파일 다운로드 실패: {url}", exc_info=True)
        return False

def scrape_post_details(post_url, board_save_dir, post_id):
    try:
        response = requests.get(post_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.select_one('.bbs_detail_tit h2')
        if not title_tag:
            logging.warning(f"  [오류] 제목을 찾을 수 없습니다: {post_url}")
            return
        title = sanitize_filename(title_tag.get_text(strip=True))
        
        folder_name = f"{post_id}_{title}"
        post_save_dir = os.path.join(board_save_dir, folder_name)
        os.makedirs(post_save_dir, exist_ok=True)

        content_tag = soup.select_one('.bbs-view-content')
        if content_tag:
            content = content_tag.get_text('\n', strip=True)
            with open(os.path.join(post_save_dir, 'content.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
        
        attachment_links = soup.select('.bbs_detail_file a[href*="fn_egov_downFile"]')
        if attachment_links:
            logging.info(f"  첨부파일 {len(attachment_links)}개 발견...")
            base_url = f"{urlparse(post_url).scheme}://{urlparse(post_url).netloc}"
            
            for i, link in enumerate(attachment_links):
                js_call = link['href']
                match = re.search(r"fn_egov_downFile\('([^']+)','([^']+)'\)", js_call)
                if match:
                    file_id, file_sn = match.groups()
                    raw_filename = link.get_text(strip=True)
                    file_name = re.sub(r'\s*\([^)]*(kb|mb|gb)[^)]*\)', '', raw_filename, flags=re.I).strip()
                    file_name = sanitize_filename(file_name)
                    download_url = urljoin(base_url, f"/cmm/fms/FileDown.do?atchFileId={file_id}&fileSn={file_sn}")
                    file_save_path = os.path.join(post_save_dir, file_name)
                    download_file(download_url, file_save_path)
    except Exception as e:
        logging.error(f"  [오류] 게시글 처리 중 예외 발생: {post_url}", exc_info=True)

# --- 로직이 대폭 수정된 scrape_board 함수 ---
def scrape_board(board_name, board_url, stop_date):
    """게시판을 순회하며 'stop_date' 이전 게시물이나 이미 다운로드한 게시물을 만나면 중단합니다."""
    logging.info(f"게시판 스크래핑 시작: {board_name} (수집 기준일: {stop_date.strftime('%Y-%m-%d')})")
    
    board_save_dir = os.path.join(BASE_SAVE_DIR, board_name)
    os.makedirs(board_save_dir, exist_ok=True)
    
    logging.info(f"'{board_save_dir}'에서 기존 다운로드된 게시글 목록 확인 중...")
    existing_post_ids = {
        fn.split('_', 1)[0] for fn in os.listdir(board_save_dir)
        if os.path.isdir(os.path.join(board_save_dir, fn)) and fn.split('_', 1)[0].isdigit()
    }
    logging.info(f"기존 게시글 {len(existing_post_ids)}개를 확인했습니다.")

    page_index = 1
    keep_scraping = True
    new_posts_found_on_board = 0

    while keep_scraping:
        paginated_url = f"{board_url}?pageIndex={page_index}"
        logging.info(f"페이지 스크래핑 중: {paginated_url}")
        time.sleep(random.uniform(0.5, 1.2))

        try:
            response = requests.get(paginated_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            posts = soup.select('.basic_table tbody tr')
            if not posts:
                logging.info("  테이블에 게시글이 전혀 없어 스크래핑을 중단합니다.")
                break
            
            found_regular_post_on_page = False
            for post in posts:
                if post.select_one('.notice_bul'):
                    logging.debug("  공지글(notice) 건너뜁니다.")
                    continue
                
                found_regular_post_on_page = True

                # --- 1. 날짜 확인 로직 ---
                date_tag = post.select_one('td:nth-of-type(4)') # 4번째 td가 날짜일 것으로 가정
                if date_tag:
                    try:
                        post_date_str = date_tag.get_text(strip=True)
                        post_date = datetime.strptime(post_date_str, '%Y-%m-%d')
                        if post_date < stop_date:
                            logging.info(f"  수집 기준일({stop_date.strftime('%Y-%m-%d')})보다 오래된 게시물(작성일: {post_date_str})을 발견하여 수집을 중단합니다.")
                            keep_scraping = False
                            break
                    except ValueError:
                        logging.warning(f"  날짜 형식을 파싱할 수 없습니다: '{date_tag.get_text(strip=True)}'. 날짜 검사를 건너뜁니다.")
                
                # --- 2. ID 확인 로직 ---
                link_tag = post.select_one('.list_subject a, .list_subject form input[type=submit]')
                if not link_tag: continue
                
                post_id = None
                if link_tag.name == 'a':
                    post_url = urljoin(board_url, link_tag['href'])
                    query_params = parse_qs(urlparse(post_url).query)
                    post_id = query_params.get('nttId', [None])[0]
                else:
                    form = link_tag.find_parent('form')
                    post_id = form.select_one('input[name="nttId"]')['value']
                    bbsId = form.select_one('input[name="bbsId"]')['value']
                    post_url = urljoin(board_url, f"{form['action']}?nttId={post_id}&bbsId={bbsId}")
                
                if not post_id:
                    logging.warning(f"  게시글의 고유 ID를 찾을 수 없어 건너뜁니다.")
                    continue

                if post_id in existing_post_ids:
                    logging.info(f"  이미 다운로드된 게시물(ID: {post_id})을 발견하여 수집을 중단합니다.")
                    keep_scraping = False
                    break

                post_title_preview = link_tag.get('value') if link_tag.name == 'input' else link_tag.get_text(strip=True)
                logging.info(f"신규 게시글 발견: {post_title_preview} (ID: {post_id})")
                scrape_post_details(post_url, board_save_dir, post_id)
                new_posts_found_on_board += 1
            
            # --- 3. 무한 루프 방지 로직 ---
            if not found_regular_post_on_page:
                logging.info("  현재 페이지에서 일반 게시물을 더 이상 찾을 수 없어 스크래핑을 종료합니다.")
                break

            if keep_scraping:
                page_index += 1

        except requests.exceptions.RequestException as e:
            logging.error(f"페이지 로드 실패: {paginated_url}", exc_info=True)
            break
        except Exception as e:
            logging.error(f"페이지 처리 중 예외 발생: {paginated_url}", exc_info=True)
            break
    
    if new_posts_found_on_board == 0:
        logging.info(f"'{board_name}' 게시판에 새로운 게시글이 없습니다.")


def main():
    """메인 실행 함수"""
    setup_logging()
    load_dotenv()
    
    logging.info("="*50)
    logging.info("학교 공지사항 크롤러를 시작합니다.")
    logging.info("="*50)

    # .env에서 수집 연도 설정 불러오기
    try:
        target_year = int(os.getenv('TARGET_YEAR', datetime.now().year - 1))
    except (ValueError, TypeError):
        target_year = datetime.now().year - 1
        logging.warning(f"TARGET_YEAR 환경변수 설정이 잘못되어 기본값(작년: {target_year})을 사용합니다.")
    
    # 수집 기준 날짜 설정 (예: 2023년 1월 1일)
    stop_date = datetime(target_year, 1, 1)

    url_string = os.getenv('SCHOOL_ANNOUNCEMENT_URLS') or os.getenv('SCHOOL_ANNOUNCEMENT_URL')
    if not url_string:
        logging.critical(".env 파일에 'SCHOOL_ANNOUNCEMENT_URLS' 또는 'SCHOOL_ANNOUNCEMENT_URL'가 설정되지 않았습니다.")
        return
    if '|' not in url_string:
        url_string = f"공지사항|{url_string}"
        
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    boards = parse_announcement_urls(url_string)
    
    for board_name, board_url in boards.items():
        scrape_board(board_name, board_url, stop_date)
        
    logging.info("="*50)
    logging.info(f"모든 작업이 완료되었습니다. 데이터는 '{BASE_SAVE_DIR}' 폴더에 저장되었습니다.")
    logging.info("로그는 'logs/crawler.log' 파일에 기록되었습니다.")
    logging.info("="*50)

if __name__ == "__main__":
    main()