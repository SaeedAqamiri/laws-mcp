# laws-mcp — ای‌ام‌سی‌پی قوانین ایران (اختبار)

سرویس MCP برای فهرست‌کردن، خواندن و دانلود قوانین از [اختبار](https://www.ekhtebar.ir/قوانین/).

## کارها

- **index** — اسکرپ صفحه «قوانین» اختبار (~۶۵۰ قانون: عنوان، لینک، سال، لینک PDF) و کش در `data/index.json`
- **list** — صفحه‌بندی روی ایندکس کش‌شده با فیلتر عنوان/سال
- **search** — جستجوی کل سایت اختبار
- **get** — گرفتن متن کامل یک قانون و ذخیره به‌صورت مارک‌داون در `data/laws/<slug>.md`
- **pdf** — دانلود PDF رسمی قانون؛ متن‌کشی با `pdftotext`؛ اگر PDF اسکن‌شده بود، OCR با VLM (GLM از Z.AI)
- **sync / sync_status** — دانلود گروهی پس‌زمینه‌ای (قابل resume؛ از وضعیت فعلی `data/` ادامه می‌دهد)
- **local** — فهرست قوانین دانلودشده روی دیسک

نکته: ابزارها فایل ذخیره می‌کنند و فقط پیش‌نمایش برمی‌گردانند؛ برای متن کامل، خود فایل را بخوان.

## راه‌اندازی

```bash
cd ~/laws-mcp && uv sync
```

در `~/.config/opencode/opencode.jsonc`:

```jsonc
"iran-laws": {
  "type": "local",
  "command": ["/home/saeed/laws-mcp/.venv/bin/python", "/home/saeed/laws-mcp/server.py"],
  "timeout": 300000
}
```

## متغیرهای محیطی (همه اختیاری)

| متغیر | پیش‌فرض | کاربرد |
|---|---|---|
| `ZAI_API_KEY` | — | برای OCR اسکن‌ها (GLM vision) |
| `LAWS_VLM_MODEL` | `glm-4.6v` | مدل VLM برای OCR |
| `LAWS_DATA_DIR` | `./data` | محل ذخیره داده‌ها |
| `LAWS_DELAY` | `0.4` | تأخیر مودبانه بین درخواست‌ها (ثانیه) |
| `FIRECRAWL_URL` | — | اگر firecrawl سلف‌هاست بالا بود، صفحه‌ها از آن گرفته می‌شوند |

## مثال استفاده (توسط ایجنت)

1. «قانون کار» → `iran-laws_get(ref="قانون کار")` → فایل مارک‌داون + مسیر
2. قوانین سال ۱۴۰۳ → `iran-laws_list(year="1403")`
3. دانلود کل قوانین → `iran-laws_sync()` و بعد `iran-laws_sync_status()`
4. PDF اسکن‌شدهٔ طولانی → `iran-laws_pdf(ref=..., max_ocr_pages=6)` و ادامه با `ocr_start_page`
