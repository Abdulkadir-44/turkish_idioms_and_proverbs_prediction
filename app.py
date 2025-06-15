from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# T5 modelini yükle (yerel klasörden)
model_path = "./training/mt5_atasozleri_checkpoints/checkpoint-1800" # eğitim yapıldıktan sonra "mt5_atasozleri_checkpoints" klasörü oluşturulacak otomatik olarak ve  içinde eğitilmiş model olacak en son checkpoint neyse onu yazacağız şimdilik checkpoint-1800 ama değişebilir
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_text = request.form["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=100)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
