from flask import Flask, render_template, request, redirect, url_for, flash
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
import torch
import nltk
import os

nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student')
def student():
    return render_template('student.html')

@app.route('/faculty')
def faculty():
    return render_template('faculty.html')

@app.route('/upload_student', methods=['POST'])
def upload_student():
    if 'answer_key' not in request.files or 'student_answers' not in request.files:
        flash('No file part')
        return redirect(request.url)

    answer_key = request.files['answer_key']
    student_answers = request.files['student_answers']

    if answer_key.filename == '' or student_answers.filename == '':
        flash('No selected file')
        return redirect(request.url)

    answer_key_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_key.filename)
    student_answers_path = os.path.join(app.config['UPLOAD_FOLDER'], student_answers.filename)

    answer_key.save(answer_key_path)
    student_answers.save(student_answers_path)

    results, final_grade = process_pdfs_student(answer_key_path, student_answers_path)

    return render_template('results_student.html', results=results, final_grade=final_grade)

@app.route('/upload_faculty', methods=['POST'])
def upload_faculty():
    if 'questions' not in request.files or 'answer_key' not in request.files or 'student_answers' not in request.files:
        flash('No file part')
        return redirect(request.url)

    questions = request.files['questions']
    answer_key = request.files['answer_key']
    student_answers = request.files['student_answers']

    if questions.filename == '' or answer_key.filename == '' or student_answers.filename == '':
        flash('No selected file')
        return redirect(request.url)

    questions_path = os.path.join(app.config['UPLOAD_FOLDER'], questions.filename)
    answer_key_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_key.filename)
    student_answers_path = os.path.join(app.config['UPLOAD_FOLDER'], student_answers.filename)

    questions.save(questions_path)
    answer_key.save(answer_key_path)
    student_answers.save(student_answers_path)

    results, final_grade = process_pdfs_faculty(questions_path, answer_key_path, student_answers_path)

    return render_template('results_faculty.html', results=results, final_grade=final_grade)

def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def calculate_similarity_with_bert(text1, text2):
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state[:, 0, :]
    embeddings2 = outputs2.last_hidden_state[:, 0, :]
    
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
    return cosine_sim

def adjust_score_for_word_count(similarity, correct_answer, student_answer):
    correct_word_count = len(correct_answer.split())
    student_word_count = len(student_answer.split())

    if student_word_count < correct_word_count:
        penalty = (correct_word_count - student_word_count) / correct_word_count
        similarity *= (1 - penalty)

    return similarity

def grade_similarity(score):
    if score > 0.9:
        return 'A'
    elif score > 0.75:
        return 'B'
    elif score > 0.6:
        return 'C'
    elif score > 0.45:
        return 'D'
    else:
        return 'F'

def calculate_final_grade(average_similarity):
    if average_similarity > 0.9:
        return 'A'
    elif average_similarity > 0.75:
        return 'B'
    elif average_similarity > 0.6:
        return 'C'
    elif average_similarity > 0.45:
        return 'D'
    else:
        return 'F'

def process_pdfs_student(answer_key_path, student_answers_path):
    correct_answers_text = extract_text_from_pdf(answer_key_path)
    student_answers_text = extract_text_from_pdf(student_answers_path)

    correct_answer_list = correct_answers_text.split("\n\n")
    student_answer_list = student_answers_text.split("\n\n")

    results = {}
    total_similarity = 0
    for i, (correct_answer, student_answer) in enumerate(zip(correct_answer_list, student_answer_list), start=1):
        similarity = calculate_similarity_with_bert(correct_answer, student_answer)
        adjusted_similarity = adjust_score_for_word_count(similarity, correct_answer, student_answer)
        total_similarity += adjusted_similarity
        grade = grade_similarity(adjusted_similarity)
        results[str(i)] = (student_answer, correct_answer, round(adjusted_similarity, 2)*100, grade)


    average_similarity = total_similarity / len(correct_answer_list)
    final_grade = calculate_final_grade(average_similarity)

    return results, final_grade

def process_pdfs_faculty(questions_path, answer_key_path, student_answers_path):
    questions_text = extract_text_from_pdf(questions_path)
    correct_answers_text = extract_text_from_pdf(answer_key_path)
    student_answers_text = extract_text_from_pdf(student_answers_path)

    correct_answer_list = correct_answers_text.split("\n\n")
    student_answer_list = student_answers_text.split("\n\n")

    results = {}
    total_similarity = 0
    for i, (correct_answer, student_answer) in enumerate(zip(correct_answer_list, student_answer_list), start=1):
        similarity = calculate_similarity_with_bert(correct_answer, student_answer)
        adjusted_similarity = adjust_score_for_word_count(similarity, correct_answer, student_answer)
        total_similarity += adjusted_similarity
        grade = grade_similarity(adjusted_similarity)
        results[str(i)] = (student_answer, correct_answer, round(adjusted_similarity, 2), grade)

    average_similarity = total_similarity / len(correct_answer_list)
    final_grade = calculate_final_grade(average_similarity)

    return results, final_grade

if __name__ == "__main__":
    app.run(debug=True)
