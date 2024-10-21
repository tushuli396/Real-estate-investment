import os
from flask import Flask, render_template

app = Flask(__name__)

# Directory where your Jupyter notebooks are stored
NOTEBOOK_DIR = 'notebooks'

@app.route('/')
def index():
    # List all notebooks in the 'notebooks' directory
    notebooks = os.listdir(NOTEBOOK_DIR)
    return render_template('index.html', notebooks=notebooks)

@app.route('/notebook/<notebook_name>')
def display_notebook(notebook_name):
    # Render the selected notebook's HTML file
    return render_template(notebook_name)

if __name__ == '__main__':
    app.run(debug=True)
