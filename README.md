# Predict_Score
Utilizing NLP techniques alongside Pandas, Spacy, NLTK, and more, the project aims to compare original and student descriptive(Essay) answers, innovating automated grading systems for educational enhancement.
### Descriptive Answer Evaluation System

#### Overview:
This repository contains modules designed for automating the evaluation of descriptive answers. The system utilizes various NLP techniques and libraries to compare original answers with student responses and predict scores based on a multitude of linguistic factors.

#### Modules:

1. **Data Loading Module**: Loads test data from an Excel file using Pandas, extracting original and student answers.

2. **Coreference Resolution Module**: Utilizes NeuralCoref and spaCy to handle coreference resolution, enhancing text clarity by replacing pronouns.

3. **Text Cleaning Module**: Cleans text by removing punctuation, converting to lowercase, fixing typos via TextBlob, and counts typos in student answers.

4. **Part-of-Speech Tagging and Lemmatization Module**: Employs NLTK for part-of-speech tagging and lemmatization to obtain base forms of words based on their parts of speech.

5. **Document-Term Matrix Module**: Utilizes scikit-learn's CountVectorizer to create a matrix representing word occurrences.

6. **Word Count and Comparison Module**: Calculates common, missing, total, and distinctly different words between original and student answers. Also, identifies proper sentences for similarity scoring.

7. **Sentence Similarity Scoring Module**: Calculates semantic similarity between sentences using spaCy based on selected nouns.

8. **Missing Noun Identification Module**: Identifies missing nouns in student answers from selected sentences and calculates similarity scores.

9. **Feature Engineering Module**: Derives additional features like missed topics fraction and new topics fraction from word and noun counts.

10. **Model Training and Prediction Module**: Trains a linear regression model using scikit-learn on training data and predicts scores based on calculated features.

11. **Output and Evaluation Module**: Displays predicted scores, evaluating student performance and highlighting areas for improvement.

#### Deployment:

This system is designed to be deployed using Streamlit. Run the application using the command:
```
streamlit run final_ui.py
```

### Notes:
- Ensure all necessary dependencies are installed as specified in the project's setup.
- The codebase contains all the required modules and functionalities for automated descriptive answer evaluation.

### OUTPUT:

![Uploding paper](
