# crawler.py

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv

# 스크래핑한 데이터를 저장할 기본 디렉토리 (README.md와 일치)
BASE_SAVE_DIR = "crawled_data"

# HTTP 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def sanitize_filename(name):
    """파일이나 폴더 이름으로 사용할 수 없는 문자를 제거하거나 변경합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def parse_announcement_urls(url_string):
    """주어진 문자열을 파싱하여 {게시판이름: URL} 딕셔너리를 반환합니다."""
    board_dict = {}
    if not url_string:
        return board_dict
    lines = url_string.strip().split(',')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            name, url = line.split('|')
            board_dict[name.strip()] = url.strip()
        except ValueError:
            print(f"잘못된 형식의 라인을 건너뜁니다: {line}")
    return board_dict

def download_file(url, save_path):
    """주어진 URL에서 파일을 다운로드하여 지정된 경로에 저장합니다."""
    try:
        with requests.get(url, headers=HEADERS, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"  [성공] 파일 다운로드: {os.path.basename(save_path)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  [오류] 파일 다운로드 실패: {url}, 에러: {e}")
        return False

def scrape_post_details(post_url, board_save_dir, post_id):
    """게시글 상세 페이지를 스크래핑하고 내용과 첨부파일을 저장합니다."""
    try:
        response = requests.get(post_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.select_one('.bbs_detail_tit h2')
        if not title_tag:
            print(f"  [오류] 제목을 찾을 수 없습니다: {post_url}")
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
            print(f"  첨부파일 {len(attachment_links)}개 발견...")
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

    except requests.exceptions.RequestException as e:
        print(f"  [오류] 게시글 페이지 로드 실패: {post_url}, 에러: {e}")
    except Exception as e:
        print(f"  [오류] 게시글 처리 중 예외 발생: {post_url}, 에러: {e}")

def scrape_board(board_name, board_url):
    """게시판의 모든 페이지를 순회하며 조건에 맞는 게시글을 스크랩합니다."""
    print(f"\n{'='*50}\n게시판 스크래핑 시작: {board_name}\n{'='*50}")
    
    board_save_dir = os.path.join(BASE_SAVE_DIR, board_name)
    os.makedirs(board_save_dir, exist_ok=True)
    
    print(f"'{board_save_dir}'에서 기존에 다운로드된 게시글 목록을 확인합니다...")
    existing_post_ids = set()
    try:
        for folder_name in os.listdir(board_save_dir):
            item_path = os.path.join(board_save_dir, folder_name)
            if os.path.isdir(item_path):
                post_id_part = folder_name.split('_', 1)[0]
                if post_id_part.isdigit():
                    existing_post_ids.add(post_id_part)
    except FileNotFoundError:
        pass
    print(f"기존 게시글 {len(existing_post_ids)}개를 확인했습니다.")

    # --- [최적화 로직 적용] ---
    page_index = 1
    keep_scraping = True
    new_posts_found_on_board = 0

    while keep_scraping:
        paginated_url = f"{board_url}?pageIndex={page_index}"
        print(f"페이지 스크래핑 중: {paginated_url}")

        try:
            response = requests.get(paginated_url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            posts = soup.select('.basic_table tbody tr')
            
            if not posts:
                print("  게시글이 더 이상 없습니다. 다음 게시판으로 넘어갑니다.")
                break
            
            for post in posts:
                # 공지글은 건너뜀
                if post.select_one('.notice_bul'):
                    continue
                
                # --- ID 추출 로직 (이전과 동일) ---
                link_tag = post.select_one('.list_subject a, .list_subject form input[type=submit]')
                if not link_tag: continue
                
                post_id = None
                if link_tag.name == 'a':
                    post_url = urljoin(board_url, link_tag['href'])
                    parsed_url = urlparse(post_url)
                    query_params = parse_qs(parsed_url.query)
                    if 'nttId' in query_params:
                        post_id = query_params['nttId'][0]
                else:
                    form = link_tag.find_parent('form')
                    action = form['action']
                    nttId = form.select_one('input[name="nttId"]')['value']
                    bbsId = form.select_one('input[name="bbsId"]')['value']
                    relative_post_url = f"{action}?nttId={nttId}&bbsId={bbsId}"
                    post_url = urljoin(board_url, relative_post_url)
                    post_id = nttId
                
                if not post_id:
                    post_title_preview = link_tag.get('value') if link_tag.name == 'input' else link_tag.get_text(strip=True)
                    print(f"  [경고] 게시글의 고유 ID(nttId)를 찾을 수 없어 건너뜁니다: {post_title_preview}")
                    continue

                # --- [핵심 최적화 로직] ---
                # ID를 확인하여 이미 다운로드된 게시물이면, 현재 게시판 크롤링을 중단
                if post_id in existing_post_ids:
                    print(f"\n  [업데이트 중단] 이미 다운로드된 게시물(ID: {post_id})을 발견했습니다.")
                    print("  이 게시판의 최신 정보 수집을 완료합니다.")
                    keep_scraping = False # 외부 while 루프를 중단시키기 위한 플래그 설정
                    break # 현재 for 루프(페이지 내 게시물 목록)를 즉시 중단

                # 새로운 게시물인 경우
                post_title_preview = link_tag.get('value') if link_tag.name == 'input' else link_tag.get_text(strip=True)
                print(f"신규 게시글 발견: {post_title_preview}")
                scrape_post_details(post_url, board_save_dir, post_id)
                new_posts_found_on_board += 1

            # for 루프가 break 없이 정상적으로 끝나면 페이지를 증가시키고, 아니면 while 루프가 종료됨
            page_index += 1

        except requests.exceptions.RequestException as e:
            print(f"  [오류] 페이지 로드 실패: {paginated_url}, 에러: {e}")
            break
        except Exception as e:
            print(f"  [오류] 페이지 처리 중 예외 발생: {paginated_url}, 에러: {e}")
    
    if new_posts_found_on_board == 0:
        print(f"\n  '{board_name}' 게시판에 새로운 게시글이 없습니다.")


def main():
    """메인 실행 함수"""
    load_dotenv()
    
    url_string = os.getenv('SCHOOL_ANNOUNCEMENT_URLS')
    if not url_string:
        url_string = os.getenv('SCHOOL_ANNOUNCEMENT_URL')
        if url_string:
            url_string = f"공지사항|{url_string}"

    if not url_string:
        print("[오류] .env 파일에 'SCHOOL_ANNOUNCEMENT_URLS' 또는 'SCHOOL_ANNOUNCEMENT_URL'가 설정되지 않았습니다.")
        print("설정 예시: SCHOOL_ANNOUNCEMENT_URLS=일반공지|https://.../notice,학사공지|https://.../academic")
        return
        
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    boards = parse_announcement_urls(url_string)
    
    for board_name, board_url in boards.items():
        scrape_board(board_name, board_url)
        
    print(f"\n{'='*50}\n모든 작업이 완료되었습니다. 데이터는 '{BASE_SAVE_DIR}' 폴더에 저장되었습니다.\n{'='*50}")

if __name__ == "__main__":
    main()