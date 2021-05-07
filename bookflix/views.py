from django.shortcuts import render
from django.http import HttpResponse
from .forms import UserForm
from .forms import UserForm2
import pandas as pd
import numpy as np
import csv
from plotly.offline import plot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pickle
from scipy.sparse import csr_matrix


# Create your views here.


def load_data(nrows):
    data = pd.read_csv("final_books3.csv")
    return data


def load_data1(nrows):
    data = pd.read_csv("reco.csv")
    return data


def load_data2(nrows):
    data = pd.read_csv("sentiments.csv")
    return data


def index(request):
    books = load_data(10000)
    original_data = books

    users = load_data1(10000)
    original_data1 = users

    reviews = load_data2(10000)
    original_data2 = reviews

    # Books Distribution
    cnt_srs = books["average_rating"].value_counts()
    cnt = cnt_srs.sort_index()
    cnt_srs1 = books["language_code"].value_counts()[:5]
    df_year = books[books['original_publication_year'] >= 1950]
    cnt_year = df_year["original_publication_year"].value_counts()
    cnt_y = cnt_year.sort_index()

    # Trending Books
    most_popular = books.sort_values("ratings_count", ascending=False)[:5]
    high_rated_book = books.sort_values('average_rating', ascending=False)[:5]
    title_pop = most_popular['original_title'].values.tolist()
    title_rate = high_rated_book['original_title'].values.tolist()
    Images_pops = most_popular['image_url'].values.tolist()
    Images_rates = high_rated_book['image_url'].values.tolist()
    ratings = high_rated_book['average_rating'].values.tolist()

    # Authors
    cross_author_counts = books['authors'].value_counts().reset_index()
    cross_author_counts.columns = ['value', 'count']
    cross_author_counts['value'] = cross_author_counts['value']
    cross_author_counts = cross_author_counts.sort_values(
        'count', ascending=False)[:5]
    count = cross_author_counts['count'].values.tolist()
    values = cross_author_counts['value'].values.tolist()

    high_rated_author = books.groupby(
        'authors')['average_rating'].mean().reset_index()
    high_rated_author.columns = ['values', 'count']
    high_rated_author = high_rated_author.sort_values(
        'count', ascending=False)[:5]
    auth = high_rated_author['values'].values.tolist()
    rate = high_rated_author['count'].values.tolist()

    # Users Distribution
    users_city = users.city.value_counts()[0:10].reset_index().rename(
        columns={'index': 'city', 'city': 'count'})
    users_state = users.state.value_counts()[0:10].reset_index().rename(
        columns={'index': 'state', 'state': 'count'})
    users_country = users.country.value_counts()[0:10].reset_index().rename(
        columns={'index': 'country', 'country': 'count'})
    cnt_age = users["Age_dist"].value_counts()
    cnt_a = cnt_age.sort_index()

    # Books and authors details
    submitbutton = request.POST.get("submit")

    field = ''
    authors_perf = ''
    title, author, language, average_rating, total_ratings, year, image, cnt1, img_popular, img_rating, pop_title, rev_title = [
    ], [], [], [], [], [], [], [], [], [], [], []
    form = UserForm()
    if request.method == 'POST':
        if 'submit' in request.POST:
            form = UserForm(request.POST)
            if form.is_valid():
                field = form.cleaned_data.get("field")
                authors_perf = form.cleaned_data.get("authors_perf")
                metadata = books[books['original_title'] == field]
                title = metadata['original_title'].values.tolist()
                author = metadata['authors'].values.tolist()
                language = metadata['language_code'].values.tolist()
                average_rating = metadata['average_rating'].values.tolist()
                total_ratings = metadata['ratings_count'].values.tolist()
                year = metadata['original_publication_year'].values.tolist()
                image = metadata['image_url'].values.tolist()

                df_auth = books[books['authors'] == author[0]]
                df_auth_sort = df_auth.sort_values(
                    'original_publication_year', ascending=False)
                cnt1 = df_auth_sort.groupby('original_publication_year')[
                    'average_rating'].mean()
                cnt1 = cnt1.reset_index()
                auth_popular = df_auth.sort_values(
                    'ratings_count', ascending=False)[:5]
                auth_high = df_auth.sort_values(
                    'average_rating', ascending=False)[:5]

                img_popular = auth_popular['image_url'].values.tolist()
                img_rating = auth_high['image_url'].values.tolist()
                pop_title = auth_popular['original_title'].values.tolist()
                rev_title = auth_high['original_title'].values.tolist()

    # Sentiment Analysis
    pol = reviews['polarity']
    sub = reviews['subjectivity']
    final = [pol, sub]
    group_labels = ['Polarity', 'Subjectivity']

    fig = ff.create_distplot(
        final, group_labels, show_hist=False)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600, width=850,
        title_font_color="#b81024",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="white"
        )
    )

    plt_div = plot(fig, output_type='div')

    cnt2 = reviews['analysis'].value_counts()
    colors = ['rgb(33,113,181)', 'fb9b06', 'rgb(65,171,93)']
    fig1 = go.Figure(data=[go.Pie(labels=cnt2.index,
                                  values=cnt2)])
    fig1.update_traces(hoverinfo='label+value', textinfo='label+percent', textfont_size=18,
                       marker=dict(colors=colors))
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=550, width=800,
        font=dict(
            size=18,
            color="white"
        ))

    plt_div1 = plot(fig1, output_type='div')

    context = {'cnt': cnt, 'cnt_y': cnt_y,
               'img_pop': Images_pops, 'title_pop': title_pop, 'Images_rates': Images_rates, 'title_rate': title_rate, 'ratings': ratings,
               'values': values, 'count': count, 'auth': auth, 'rate': rate, 'users_city': users_city, 'cnt_a': cnt_a, 'users_state': users_state, 'users_country': users_country,
               'form': form,   'submitbutton': submitbutton,   'title': title, 'author': author,
               'language': language, 'average_rating': average_rating, 'total_ratings': total_ratings, 'year': year, 'image': image, 'authors_perf': authors_perf,
               'img_popular': img_popular, 'img_rating': img_rating, 'pop_title': pop_title, 'rev_title': rev_title, 'cnt1': cnt1,
               'plt_div': plt_div, 'plt_div1': plt_div1}
    return render(request, 'index.html', context)


def Home(request):
    return render(request, 'home.html')


def recommend(request):
    books = load_data(10000)
    original_data = books

    users = load_data1(10000)
    original_data1 = users

    # Content-based Recommendation
    content_data = books[['original_title',
                          'authors', 'average_rating', 'image_url']]
    content_data = content_data.astype(str)

    content_data['content'] = content_data['original_title'] + ' ' + content_data['authors'] + \
        ' ' + content_data['average_rating'] + ' ' + content_data['image_url']

    content_data = content_data.reset_index()
    indices = pd.Series(content_data.index,
                        index=content_data['original_title'])

    with open('sim_score.pkl', 'rb') as f:
        sim_score = pickle.load(f)

    def get_recommendations(title, cosine_sim=sim_score):
        idx = indices[title]

        # Get the pairwsie similarity scores of all books with that book
        sim_scores = list(enumerate(sim_score[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 5 most similar books
        sim_scores = sim_scores[1:6]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar books
        return list(content_data['image_url'].iloc[book_indices])

    # collaborative recommendation
    book_pivot = users.pivot_table(
        columns='user_id', index='book_title', values='rating')
    book_pivot.fillna(0, inplace=True)
    book_sparse = csr_matrix(book_pivot)
    loaded_model = pd.read_pickle('collaborative_model.pkl')

    def reco(book_name):
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, suggestions = loaded_model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1))
        return book_pivot.index[suggestions]

    submitbutton = request.POST.get("submit")
    select5 = ''
    Images_array = []
    captions = []
    ratings = []
    rec = []
    img_list = []
    rate_list = []
    form = UserForm2()
    if request.method == 'POST':
        if 'submit' in request.POST:
            form = UserForm2(request.POST)
            if form.is_valid():
                select5 = form.cleaned_data.get("select5")
                book_rec = get_recommendations(select5, sim_score)
                for book in book_rec:
                    Images_array.append(book)
                    name = books[books['image_url'] ==
                                 book]['original_title'].values[0]
                    rate = books[books['image_url'] ==
                                 book]['average_rating'].values[0]
                    captions.append(name)
                    ratings.append(rate)
                book = reco(select5)
                rec = book.tolist()
                rec = rec[0]
                for i in rec:
                    url = users[users['book_title'] == i]['img_l'].values[0]
                    img_list.append(url)

    context = {'form': form, 'submitbutton': submitbutton, 'Images_array': Images_array,
               'captions': captions, 'img_list': img_list, 'rec': rec}

    return render(request, 'recommend.html', context)
