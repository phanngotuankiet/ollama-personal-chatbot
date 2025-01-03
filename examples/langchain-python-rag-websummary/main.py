# đây là tính năng đọc nội dung từ website thông qua url và tóm tắt nội dung
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import OllamaLLM

# Đoạn này đọc nội dung từ website
loader = WebBaseLoader("https://ollama.com/blog/run-llama2-uncensored-locally")
docs = loader.load()
# WebBaseLoader làm những điều sau:
# 1. Sử dụng BeautifulSoup4 để tải nội dung HTML từ url
# 2. parse HTML để lấy text content
# 3. Loại bỏ các thẻ HTML, scripts JS, css
# 4. Trả về nội dung text sạch

llm = OllamaLLM(model="codellama") # khởi tạo model thông qua ollama server local
chain = load_summarize_chain(llm, chain_type="stuff") # tạo chain chuyên dụng để tóm tắt nội dung

result = chain.invoke(docs) # thực hiện tóm tắt: 1. Lấy nội dung từ 'docs', 2. Tạo prompt lên model LLM yêu cầu tóm tắt
print(result) # in kết quả

# chain_type="stuff": Đơn giản nhất - ghép tất cả văn bản thành một prompt và gọi LLM một lần để tóm tắt. 
# Phù hợp với văn bản ngắn.

# chain_type="refine": Tóm tắt từng phần văn bản và tinh chỉnh dần. Đầu tiên tóm tắt phần đầu, 
# sau đó dùng kết quả đó để tóm tắt tiếp phần sau. Phù hợp với văn bản dài và cần độ chính xác cao.

# chain_type="map_reduce": Chia văn bản thành nhiều phần nhỏ, tóm tắt từng phần (map), 
# sau đó kết hợp các tóm tắt lại (reduce). Phù hợp với văn bản rất dài và có thể chạy song song.

# chain_type="map_rerank": Tương tự map_reduce nhưng thêm bước xếp hạng các tóm tắt để chọn ra
# những phần quan trọng nhất. Phù hợp khi cần tập trung vào những điểm chính của văn bản.