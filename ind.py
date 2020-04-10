from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

app = Flask(__name__,template_folder='template')

def weighted_rating(x,m,C,max1):
    v = x['vote_count']
    R = x['vote_average']
    popularity=x['popularity']
    return (((v/(v+m) * R) + (m/(m+v) * C)) * 0.7) + (0.3 * (popularity/max1)*10)


@app.route('/')
def home():
    
    with open('./qualified.pickle', 'rb') as f:
        qualified = pickle.load(f)

    m=434.0
    C=5.244896612406511
    max1=547.4882980000001
    
    data=qualified
    popularity=data
    
    data['wr'] = qualified.apply(weighted_rating,args=(m,C,max1,), axis=1)
    
    data = data.sort_values('wr', ascending=False).head(15)
    
    popularity = popularity.sort_values('popularity', ascending=False).head(15)
    
    #qualified=qualified.head(15)
    #pop=popularity.to_dict() 

    return render_template('index.html',l = np.array(popularity['title']))


if __name__ == '__main__':
	app.run(debug=False)