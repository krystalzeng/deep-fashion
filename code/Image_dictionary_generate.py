import pickle as pickle
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as  np
import json
from collections import Counter
from itertools import product, combinations

def read_data_desc(filename):
	data = []
	for line in open(filename, 'r'):
		data.append(json.loads(line))
	output = pd.DataFrame({'group_url': [data[x]['group_url'] for x in range(0,len(data))],
						 'image_desc':[data[x]['image_desc'] for x in range(0,len(data))],
						 'set_likes':[data[x]['set_likes'] for x in range(0,len(data))],
						 'image_url':[data[x]['image_url'] for x in range(0,len(data))],
						 'meta_views':[data[x]['meta_views'] for x in range(0,len(data))]})
	return output

def read_data(filename):
	data = []
	for line in open(filename, 'r'):
		data.append(json.loads(line))
	output_desc = pd.DataFrame({'image_urls': [data[x]['image_urls'] for x in range(0,len(data))],
						 'images':[data[x]['images'] for x in range(0,len(data))]})
	return output_desc

def filter_image(data):
	output = data
	# remove null
	output=output[output.image_desc.isnull()==False]

	# convert item descs to lower case
	output.image_desc = output.image_desc.str.lower()
	# filtering out non-clothes images

	# filter list
	clothes_df = pd.read_json('items.json', orient='index')
	clothes_df = clothes_df.reset_index()
	clothes_df.rename(columns={'index': 'title'}, inplace=True)
	len(clothes_df)

	clothing_items_list = clothes_df['title'].tolist()

	# create item and category fields
	output['item'] = ''
	output['item_category'] = ''

	filtered_df = pd.DataFrame(index=output.index, columns=output.columns)
	filtered_df.dropna(inplace=True)

	for item in clothing_items_list:
		item_filter = output[output['image_desc'].str.contains(item)]
		item_filter['item'] = item
		item_filter['item_category'] = clothes_df[clothes_df['title'] == item]['category'].to_string(index=False)
		filtered_df = pd.concat((filtered_df, item_filter), axis=0)

	return filtered_df

def find_views(x):
	start=x.find(' ago.')+5
	end=x.find('view')
	x = x[start:end]
	x = x.replace(',','')
	x = x.replace('One','1')
	x = x.replace('Two','2')
	x = x.replace('Three','3')
	x = x.strip()
	return x

def convert_word_into_number(word):
	word = word.replace(',','')
	word = word.replace('Like','1')
	word = word.strip()
	return word

def convert_dates_into_number(dates):
	converted_dates = []
	for date in dates:
		converted_dates.append(convert_date_into_num(date))
	return converted_dates

def convert_date_into_num(date):
	date = date.lower()
	total_days_elapsed = 0
	day = 'day'
	days = 'days'
	month = 'month'
	months = 'months'
	year = 'year'
	years = 'years'
	one = 'one'
	two = 'two'
	three = 'three'
	four = 'four'
	five = 'five'
	six = 'six'
	seven = 'seven'
	eight = 'eight'
	nine = 'nie'
	ten = 'ten'
	eleven = 'eleven'
	twelve = 'twelve'
	
	if days in date:
		date = date.replace(' days','')
		total_days_elapsed += int(date)
	elif day in date:
		total_days_elapsed += 1
	elif months in date:
		if two in date:
			total_days_elapsed += 60
		if three in date:
			total_days_elapsed += 90
		if four in date:
			total_days_elapsed += 120
		if five in date:
			total_days_elapsed += 150
		if six in date:
			total_days_elapsed += 180
		if seven in date:
			total_days_elapsed += 210
		if eight in date:
			total_days_elapsed += 240
		if nine in date:
			total_days_elapsed += 270
		if ten in date:
			total_days_elapsed += 300
		if eleven in date:
			total_days_elapsed += 330
		if twelve in date:
			total_days_elapsed += 360
	elif month in date:
		total_days_elapsed += 30
	
	return total_days_elapsed
	
def find_time_elapsed(x):
	end = x.find(' ago.') - 1
	start = end
	date = ''
	for i in range(start, 0, -1):
		if x[i] != '>':
			date += x[i]
		else:
			break
	return date[::-1]

# This function returns image url with paths
def generate_image_path(image_paths):
	urls = image_paths.image_urls.values
	paths = image_paths.images.values
	img_path_dict = defaultdict()
	for i in range(0, len(urls)):
		img_path_dict[urls[i][0]] = paths[i][0]
	return img_path_dict


# This function returns image url with distinct groups, and all infos
def generate_image_dict(filtered_images, image_paths):
	filtered_df = filtered_images
	# Get all the unique image urls
	unique_img_url = set(filtered_df['image_url'].values)
	# Dictionary for necessary information one image has including number of groups it is in,
	# group urls, views for each group, likes for each group, time elapsed since the group is upload,
	# as well as the percentage between likes and views, time elapsed and views
	img_info_dict = defaultdict()
	# Dictionary for image group url, key word: image url
	img_group_url_dict = defaultdict(set)

	for i in range(0, len(filtered_df)):
		image_url = filtered_df['image_url'].values[i]
		group_url = filtered_df['group_url'].values[i]
		set_likes = filtered_df['set_likes'].values[i]
		meta_views = find_views(filtered_df['meta_views'].values[i])
		time_elapsed = find_time_elapsed(filtered_df['meta_views'].values[i])
		if image_url not in img_info_dict:
			img_info_dict[image_url] = defaultdict()
			img_info_dict[image_url][group_url] = defaultdict()
			
		elif group_url not in img_info_dict[image_url]:
			img_info_dict[image_url][group_url] = defaultdict()
		img_info_dict[image_url][group_url]['Meta views'] = meta_views
		img_info_dict[image_url][group_url]['Likes'] = convert_word_into_number(set_likes)
		img_info_dict[image_url][group_url]['Time elapsed'] = convert_date_into_num(time_elapsed)
		img_group_url_dict[image_url].add(group_url)

	for unique_img in unique_img_url:
		img_info_dict[unique_img]['path'] = image_paths[unique_img]['path']
		for group in img_group_url_dict[unique_img]:
			likes = float(img_info_dict[unique_img][group]['Likes'])
			views = float(img_info_dict[unique_img][group]['Meta views'])
			time_elapsed = float(img_info_dict[unique_img][group]['Time elapsed'])
			percentage = likes / views * 100.0
			# Assuming the boundary is 100 views per day
			percentage2 = views / (time_elapsed*100) * 100.0
			percentage = 100.0 if percentage > 100.0 else percentage
			img_info_dict[unique_img][group]['Likes vs. Views (%)'] = percentage
			img_info_dict[unique_img][group]['Views vs. Time elapsed (%)'] = percentage2

	return img_group_url_dict, img_info_dict

