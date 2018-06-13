import pandas as pd
import numpy as np
import difflib as dl
import sys

def get_close(x):
	if len(x) == 0:
		return ""
	return x[0]

list_file = sys.argv[1]
rating_file = sys.argv[2]
output_file = sys.argv[3]

movie_list = open(list_file).read().splitlines()
movie_data = pd.DataFrame({'movie': movie_list})
rating_data = pd.read_csv(rating_file)
rating_data['rating'] = rating_data['rating'].astype(str).astype(float)
rating_data['counts'] = pd.Series(1, index=rating_data.index)
rating_data = rating_data.groupby(['title'])['counts', 'rating'].sum().reset_index()
rating_data['average_rating'] = pd.Series(rating_data['rating']/rating_data['counts'], index=rating_data.index)

movie_data['closed'] = pd.Series(movie_data['movie'], index=movie_data.index)
movie_data['closed'] = movie_data['closed'].apply(lambda x: dl.get_close_matches(x, rating_data['title'], n=1))
movie_data['closed'] = movie_data['closed'].apply(get_close)

result = movie_data.set_index('closed').join(rating_data.set_index('title')).reset_index()

result['average_rating'] = result['average_rating'].apply(lambda x: round(x, 2))
result = result.drop(['closed', 'rating', 'counts'], axis=1)
result = result.set_index('movie')

result.to_csv(output_file, sep=',', encoding='utf-8')
