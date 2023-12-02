import streamlit as st
import pandas as pd
import joblib
import re
import string



def predict_final_score(original_answer_script, student_answer_script):
    # assign data of lists.  
    data = {'Person': ['Teacher', 'student'], 'text': [original_answer_script, student_answer_script]}  
    data_df = pd.DataFrame(data) #data_df is unprocessed dataset.

    # Apply a first round of text cleaning techniques


    def clean_text_round1(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = text.replace(' “' , " ")
        text = text.replace("’s ", " ")
        text = text.replace('” ', " ")
        return text
        

    text_clean_1 = clean_text_round1(original_answer_script)
    text_clean_2 = clean_text_round1(student_answer_script)

    #fix typos
    from textblob import TextBlob
    
    #correct typos in students answer
    textBlb = TextBlob(text_clean_2)            
    text_fixed_typos_2 = textBlb.correct() 

    #correct typos in original answer (typos acc to textblob)
    textBlb = TextBlob(text_clean_1)            
    text_fixed_typos_1 = textBlb.correct() 

    # checking the typos not in the original text
    print(1)
    typos_list = []
    
    typos_list_corrected = []
    wordslist_original = list(text_clean_1.split())
    wordslist_1 = list(text_clean_2.split())
    wordslist_2 = list(text_fixed_typos_2.split())

    for i in range(len(wordslist_1)) :
        if wordslist_1[i] != wordslist_2[i] :
            if wordslist_1[i] not in wordslist_original :
                typos_list.append(wordslist_1[i])
                typos_list_corrected.append(wordslist_2[i])

    typos = len(typos_list)
    cleaned_1 = str(text_fixed_typos_1)
    cleaned_2 = str(text_fixed_typos_2)


    #pos tagging
    
    # Define function to lemmatize each word with its POS tag
    import nltk
    import nltk.corpus
    from nltk.corpus import wordnet

    # POS_TAGGER_FUNCTION 
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None
        

    # tokenize the sentence and find the POS tag for each token
    pos_tagged_1 = nltk.pos_tag(nltk.word_tokenize(cleaned_1))
    pos_tagged_2 = nltk.pos_tag(nltk.word_tokenize(cleaned_2))
    
    # the above pos tags are a little confusing.
    
    # we can use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged_1 = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged_1))
    wordnet_tagged_2 = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged_2))

    data_tagged = {'Person': ['Teacher', 'student'], 'text': [wordnet_tagged_1, wordnet_tagged_2]}  
    data_df_tagged = pd.DataFrame(data_tagged) 
    # data_df_tagged

    #lemmatization
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_text_1 = []
    lemmatized_text_2 = []


    for word, tag in wordnet_tagged_1:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_text_1.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_text_1.append(lemmatizer.lemmatize(word, tag))
    lemmatized_text_1 = " ".join(lemmatized_text_1)

    for word, tag in wordnet_tagged_2:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_text_2.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_text_2.append(lemmatizer.lemmatize(word, tag))
    lemmatized_text_2 = " ".join(lemmatized_text_2)
    
    data_lemmatized = {'Person': ['Teacher', 'student'], 'text': [lemmatized_text_1, lemmatized_text_2]}  
    data_df_lemmatized = pd.DataFrame(data_lemmatized) 
    # data_df_lemmatized

    from nltk.corpus import wordnet

    original_word_list = lemmatized_text_1.split()
    student_word_list = lemmatized_text_2.split()

    replaced_words_list = []

    for original_word in original_word_list:                           #replace synonyms in original_text
        synonyms = []
        for syn in wordnet.synsets(original_word):                         
            for l in syn.lemmas():
                synonyms.append(l.name())
        for synonym in synonyms:
            if (synonym in original_word_list) and (synonym != original_word):
                for i in range(len(original_word_list)):
                    if original_word_list[i] == synonym:
                        original_word_list[i] = original_word
                       
                replaced_words_list.append([synonym, original_word])
    lemmatized_text_1 = " ".join(original_word_list)



    for original_word in original_word_list:                           #replace synonyms in student text with original text
        synonyms = []
        for syn in wordnet.synsets(original_word):                         
            for l in syn.lemmas():
                synonyms.append(l.name())
        for synonym in synonyms:
            if (synonym in student_word_list) and (synonym != original_word):
                for i in range(len(student_word_list)):
                    if student_word_list[i] == synonym:
                        student_word_list[i] = original_word
                        
                replaced_words_list.append([synonym, original_word])

    for word in student_word_list:
        synonyms = []
        for syn in wordnet.synsets(word):                         
            for l in syn.lemmas():
                synonyms.append(l.name())
        for synonym in synonyms:
            if (synonym in original_word_list) and (synonym != word):
                for i in range(len(student_word_list)):
                    if student_word_list[i] == word:
                        student_word_list[i] = synonym
                replaced_words_list.append([word, synonym])
    lemmatized_text_ = " ".join(student_word_list)
            


    synonyms = []
    for syn in wordnet.synsets("story"):                         
        for l in syn.lemmas():
            synonyms.append(l.name())

    # assign data of lists.  
    data = {'Person': ['Teacher', 'student'], 'text': [lemmatized_text_1, lemmatized_text_2]}  
    data_df_cleaned = pd.DataFrame(data) #data_df is unprocessed dataset.

    # Let's take a look at our dataframe
    # data_df_cleaned

    # We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(data_df_cleaned.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names_out())
    data_dtm.index = data_df_cleaned.index
    # data_dtm

    # Counting number of common words between original and student text
    words = [ key for key in dict(data_dtm.iloc[0])] 

    common_word_count = 0
    total_words_student_count = 0
    missing_words_count = 0
    total_words_original_count = 0

    common_words = []
    total_words_student = []
    missing_words = []
    total_words_original = []

    for word in words:
        list_ = list(data_dtm[word])
        if (list_[0] != 0) and (list_[1] != 0) :
            common_word_count += 1
            common_words.append(word)
        if (list_[1] != 0) :
            total_words_student_count += 1
            total_words_student.append(word)
        if (list_[1] == 0) and (list_[0] != 0):
            missing_words_count += 1
            missing_words.append(word)
        if (list_[0] != 0) :
            total_words_original_count += 1   
            total_words_original.append(word)

    original_total_word_count = data_dtm.sum(axis = 1)[0] 
    student_total_word_count = data_dtm.sum(axis = 1)[1]
            
    common_word_count_percentage = common_word_count/ total_words_student_count

    #create lists of  nouns from lemmatized text to select proper sentences while comparing similarity
    nouns_original = []

    pos_tagged_1 = nltk.pos_tag(nltk.word_tokenize(lemmatized_text_1))
    
    # we can use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged_1 = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged_1))

    for word,tag in wordnet_tagged_1 :
        if tag == "n":
            nouns_original.append(word)

    #removing duplicates from list
    distinct_nouns_original = []
    [distinct_nouns_original.append(x) for x in nouns_original if x not in distinct_nouns_original]
    nouns_original = distinct_nouns_original

    # Getting sentences from original text and student text
    student_text_sentences_list = []
    original_text_sentences_list = []

    student_text_sentence = ""
    original_text_sentence = ""
    word_count = 0
    sentence_count = 0

    list_1 = original_answer_script.split(".")
    if list_1[len(list_1) - 1] == " ":
        list_1.pop()

    # generating original_text_sentences_list
    for word in lemmatized_text_1.split(" "):
        if word_count < len(list_1[sentence_count].lstrip().split(" ")) :
            original_text_sentence = original_text_sentence + " " + word
            word_count += 1
        else :
            word_count = 0
            sentence_count += 1
            original_text_sentences_list.append(original_text_sentence.lstrip())
            original_text_sentence = ""
            original_text_sentence = original_text_sentence + " " + word
            word_count += 1
            
    original_text_sentences_list.append(original_text_sentence.lstrip())


    # generating student_text_sentences_list
    list_2 = student_answer_script.split(".")

    if list_2[len(list_2) - 1] == " ":
        list_2.pop()
        
    word_count = 0
    sentence_count = 0

    for word in lemmatized_text_2.split(" "):
        if word_count < len(list_2[sentence_count].lstrip().split(" ")) :
            student_text_sentence = student_text_sentence + " " + word
            word_count += 1
        else :
            word_count = 0
            sentence_count += 1
            student_text_sentences_list.append(student_text_sentence.lstrip())
            student_text_sentence = ""
            student_text_sentence = student_text_sentence + " " + word
            word_count += 1
            
    student_text_sentences_list.append(student_text_sentence.lstrip())      


    # creating a class that gives nouns in a sentence
    class pos_sentence:
        def __init__(self, sentence):
            self.nouns = [noun for noun in nouns_original if noun in sentence.split(" ")]

    import spacy
    nlp = spacy.load("en_core_web_md")

    similar_sentence_list = []
    similarity_score_list = []
    missing_noun_list = []

    for sentence in original_text_sentences_list :
        similar_sentences = []
        similarity_score_overall = 0
        
        sentence_object = pos_sentence(sentence)
        distinct_nouns_sentence = []
        [distinct_nouns_sentence.append(x) for x in sentence_object.nouns if x not in distinct_nouns_sentence]
        if len(distinct_nouns_sentence) != 0 :
            for noun in distinct_nouns_sentence:
                similarity_score = 0
                for sentence_1 in student_text_sentences_list :
                    if noun in sentence_1.split(" ") :
                        if nlp(sentence_1).similarity(nlp(sentence)) > similarity_score :
                            similarity_score = nlp(sentence_1).similarity(nlp(sentence))
                            similar_sentence = sentence_1

                if similarity_score == 0:
                    missing_noun_list.append(noun)

                elif similar_sentence not in similar_sentences :
                    similar_sentences.append(similar_sentence)
                    similarity_score_overall += similarity_score
        if len(similar_sentences) != 0:
            similarity_score_list.append(similarity_score_overall/len(similar_sentences))
            similar_sentence_list.append([sentence, similar_sentences])


    nlp = spacy.load("en_core_web_md")


    distinct_nouns_original_count = len(distinct_nouns_original)

    missing_nouns_count = len(missing_noun_list)

    similarity_score = sum(similarity_score_list)/len(similarity_score_list)


    # typos_list


    #Features for model are derived as follows: 

    #Feature 1
    similarity_score = similarity_score   #considers the presence of synonyms, antonyms and wrong mapping of verbs, adj, adv of corresponding nouns

    # missing_nouns_count                   # total no of missing nouns(distinct missing topics) in student's text
    # distinct_nouns_original_count         # total no of nouns (distinct topics) in original text
    #Feature 2
    fraction_of_topics_missed = (missing_nouns_count/distinct_nouns_original_count)

    # missing_words_count                   # total no of missing words(distinct) in student's text (verbs,nouns, adjectives etc)
    # total_words_student_count             # total no of words(distinct) in student's text (excluding stop words)
    # total_words_original_count            # total no of words(distinct) in original text (excluding stop words)
    #Feature  3
    fraction_of_new_topics = (total_words_student_count -total_words_original_count + missing_words_count)/ total_words_original_count

    # model
    predicted_score =  -51.51129328 + (169.05178094 * similarity_score) + (-69.15099897*fraction_of_topics_missed) +(-20.09644363* fraction_of_new_topics)

    if predicted_score>100: #For the cases where no topics are missing and less synonyms have been used.
        predicted_score=100
                

    #here's the predicted score of given test data in the beginning
    #print(predicted_score)
    return predicted_score


 # Load the pre-trained ML model
# model = joblib.load("model.pkl")

    # Make predictions


def main():
    st.title("ML Model for Text Score Prediction")
            
    uploaded_files = st.file_uploader("Upload text file(s)", accept_multiple_files=True, type=['txt'])

    if len(uploaded_files) == 2:
        original_answer_script= uploaded_files[0].read().decode("utf-8")
        student_answer_script= uploaded_files[1].read().decode("utf-8")

    if st.button("Predict"):
        
            # Predict scores
            predicted_score = predict_final_score(original_answer_script,student_answer_script)

            # Display the scores
            st.write("Predicted Score:", predicted_score)
           
        
            if predicted_score >= 90.0:
                grade = "O"
                grade_point = 10
            elif predicted_score >= 80.0:
                grade = "A+"
                grade_point = 9.0
            elif predicted_score >= 70.0:
                grade = "A"
                grade_point = 8.0
            elif predicted_score >= 60.0:
                grade = "B+"
                grade_point = 7.0
            elif predicted_score >= 50.0:
                grade = "B"
                grade_point = 6.0
            elif predicted_score >= 40.0:
                grade = "C"
                grade_point = 5.0
            elif predicted_score >= 30.0:
                grade = "P"
                grade_point = 4.0
            else:
                grade = "F"
                grade_point = 0

            # Calculate the CGPA based on the grade point
            if grade_point >= 9.0:
                cgpa = 9.0
            elif grade_point >= 8.0:
                cgpa = 8.0
            elif grade_point >= 7.0:
                cgpa = 7.0
            elif grade_point >= 6.0:
                cgpa = 6.0
            elif grade_point >= 5.0:
                cgpa = 5.0
            elif grade_point >= 4.0:
                cgpa = 4.0
            else:
                cgpa = 0

        # Display the predicted grade and CGPA
            st.write("Predicted Grade:", grade)
            st.write("Predicted CGPA:", cgpa)

        
            
          

        # Calculate the grade based on the predicted score
            
                

if __name__ == '__main__':
    main()
