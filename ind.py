    from flask import Flask,render_template,url_for,request,redirect
    import pandas as pd 
    import numpy as np
    import pickle
    
    import mysql.connector
    
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet
    from surprise import Reader, Dataset, SVD
    import warnings; warnings.simplefilter('ignore')
    from surprise.model_selection import train_test_split,cross_validate,KFold
    from surprise import Dataset,accuracy
    
    
    app = Flask(__name__,template_folder='template')
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    name="abc"
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == "POST":
            details = request.form
            uname = "kalppn"#details['username']
            passs = "12345" #details['password']
            
            mydb = mysql.connector.connect(
              host="127.0.0.1",
              user="root",
              passwd="12345",
              database="moviezflix"
            )
    
            mycursor = mydb.cursor()
            mycursor.execute("SELECT * FROM login")        
            myresult = mycursor.fetchall()        
            mycursor.close()
            
            for x in myresult:
                
                if x[2]==uname and x[3]==passs:
                    print("Logged in")
                    details=x
                    idd=x[0]
                    name=x[1]
                    title=x[4]
                    
                    with open('./smd.pickle', 'rb') as f:
                        smd = pickle.load(f)
                    
                    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
                    count_matrix = count.fit_transform(smd['soup'])
                    
                    cosine_sim = cosine_similarity(count_matrix, count_matrix)
                    
                                 
                    smd = smd.reset_index()
                    
                    with open('./idmap.pickle', 'rb') as f:
                        id_map = pickle.load(f)
                        
                    indices_map = id_map.set_index('id')
                    
                    with open('./indices.pickle', 'rb') as f:
                        indices = pickle.load(f)
                        
                    idx = indices[title]
                    tmdbId = id_map.loc[title]['id']
                    #print(idx)
                    movie_id = id_map.loc[title]['movieId']
                    
                    sim_scores = list(enumerate(cosine_sim[int(idx)]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    sim_scores = sim_scores[1:26]
                    movie_indices = [i[0] for i in sim_scores]
                    
                    svd = SVD()
                    
                    svd = pickle.load(open("svd.p", "rb"))
                    
                    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id','tagline']]
                    movies['est'] = movies['id'].apply(lambda x: svd.predict(idd, indices_map.loc[x]['movieId']).est)
                    movies = movies.sort_values('est', ascending=False)
                    
                    """
                    mv=[]
                    mv1=[i for i in np.array([[title]])]
                    
                    mv2=[i for i in np.array(movies[['title']][0:13])]
                    mv1.extend(mv2)
                    #print(mv1)
                    """
                    
                    with open('./qualified.pickle', 'rb') as f:
                        qualified = pickle.load(f)
                        
                    qualified = qualified.head(13)
                    
                    return render_template('index1.html',l1 = np.array(movies[['title','tagline']][0:13]),l=name,l2=np.array(qualified[['title','tagline']]),l3=title)
    
            with open('./qualified.pickle', 'rb') as f:
                qualified = pickle.load(f)
                
            qualified = qualified.head(13)
            
            return render_template('index.html',l1 = np.array(qualified[['title','tagline']][0:]))
            
    @app.before_request
    def before_request():
        if 'localhost' in request.host_url or '0.0.0.0' in request.host_url:
            app.jinja_env.cache = {}
            
    def weighted_rating(x,m,C,max1):
        v = x['vote_count']
        R = x['vote_average']
        #popularity=x['popularity']
        return ((v/(v+m) * R) + (m/(m+v) * C))  # * 0.5) + (0.5 * (popularity/max1)*10)
    
    
   
    @app.route('/genre/<gen>')
    def genre_build(gen, percentile=0.85):
        with open('./gen_md.pickle', 'rb') as f:
            gen_md = pickle.load(f)
        gen=gen.title()
        df = gen_md[gen_md['genre'] == gen]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)
        
        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull() & df['popularity'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity','tagline']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('float')
        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(25)
        print(qualified.head())
        
        return render_template('genre.html',l1 = np.array(qualified[['title','tagline']][1:]),l2=gen,l=name)
    
    
    
    @app.route('/')
    def home(): 
        
        with open('./qualified.pickle', 'rb') as f:
                qualified = pickle.load(f)
        
        m=434.0
        C=5.244896612406511
        max1=547.4882980000001
                
        qualified['wr'] = qualified.apply(weighted_rating,args=(m,C,max1,), axis=1)  
                
        qualified = qualified.sort_values('wr', ascending=False).head(13)
            
        return render_template('index.html',l1 = np.array(qualified[['title','tagline']][0:]))
    
    if __name__ == '__main__':
    	app.run(debug=False)