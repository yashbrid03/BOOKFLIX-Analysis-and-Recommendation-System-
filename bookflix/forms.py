from django import forms

import pandas as pd
import csv


def load_data(nrows):
    data = pd.read_csv("final_books3.csv")
    return data


def load_data1(nrows):
    data = pd.read_csv("reco.csv")
    return data


books = load_data(10000)
original_data = books

users = load_data1(10000)
original_data1 = users


def get_my_choices():
    list1 = books['original_title'].unique().tolist()
    list2 = books['original_title'].unique().tolist()
    merged_list = tuple(zip(iter(list1), iter(list2)))
    return merged_list


def get_list():
    titles = users['book_title'].unique()
    title = books[books['original_title'].isin(titles)]
    title1 = title['original_title'].unique()
    title2 = title['original_title'].unique()
    merged_list = tuple(zip(iter(title1), iter(title2)))
    return merged_list


class UserForm(forms.Form):
    field = forms.ChoiceField(label='Select Book', choices=get_my_choices())
    CHOICES = (('Most Popular Books', 'Most Popular Books'),
               ('Highest Rated Books', 'Highest Rated Books'), ('Over The Year Performance', 'Over The Year Performance'))
    authors_perf = forms.ChoiceField(
        label='Select author Performance', choices=CHOICES)


class UserForm2(forms.Form):
    select5 = forms.ChoiceField(label='Select Book', choices=get_list())
