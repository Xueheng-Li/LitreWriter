# 📚 Literature Review Generator

`litre_writer.py` is a Python script designed to automate the process of generating literature reviews from a collection of research papers in PDF format. It uses OpenAI compatible language models to summarize papers, compile reviews, and translate them into Chinese. 🤖

## 🛠️ Setup

### ✅ Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- An OpenAI API key

### 📥 Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Xueheng-Li/LitreWriter.git
    cd LitreWriter
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory of the project and add your OpenAI compatible API key and other configurations:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    BASE_URL=your_openai_api_base_url
    MODEL_NAME=your_model_name
    ```

## ⚙️ How It Works

The script follows these steps to generate a literature review:

1. **📄 Load PDF Files**: It lists all PDF files in the specified folder.
2. **📝 Summarize Papers**: For each paper, it generates a detailed summary using OpenAI's language model.
3. **📊 Compile Review**: It compiles all summaries into a comprehensive literature review.
4. **🔄 Translate Review**: Optionally, it translates the review into Chinese.
5. **💾 Save Review**: The final review and summaries are saved as markdown files.

### 🧩 Script Components

- **📂 PDF Loading**: The `load_pdf` function loads the content of a PDF file.
- **📋 Summary Generation**: The `summary_call` function generates summaries for each paper.
- **📑 Review Compilation**: The `review_writer` function compiles the summaries into a literature review.
- **🌐 Translation**: The `translator` function translates the review into Chinese.
- **💾 File Saving**: The `save_review` function saves the review and summaries to markdown files.

## 🚀 Usage

To run the script, use the following command:
```sh
python litre_writer.py --paper_folder path/to/papers --topic "Your Research Topic"
```

### ⚡ Arguments

- `--paper_folder`: Path to the folder containing the PDF files of the research papers.
- `--topic`: The research topic for which the literature review is being generated.

### 🔍 Example

```sh
python litre_writer.py --paper_folder papers --topic "Information Acquisition in Social Networks"
```

This command will generate a literature review for all papers in the `papers` folder related to the topic "Information Acquisition in Social Networks".

## 📤 Output

The script will generate markdown files containing the literature review and summaries. These files will be saved in the specified paper folder.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
