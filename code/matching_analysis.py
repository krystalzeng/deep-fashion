import numpy as np
import random

class matching_analysis():

	def __init__(self, img_group_dict, img_dict, unique_urls):
		self.img_info_dict = img_dict
		self.img_group_url_dict = img_group_dict
		self.unique_image_urls = list(unique_urls)
		self.views_per_day = 100.0
		self.likes_per_day = 50.0
		self.average_lv = 0.0
		self.average_lt = 0.0
		self.average_vt = 0.0

	def if_in_same_group(self, img1, img2):
		groups_for_img1 = self.img_group_url_dict[img1]
		groups_for_img2 = self.img_group_url_dict[img2]
		for group in groups_for_img1:
			if group in groups_for_img2:
				return True
		return False

	def if_in_same_category(self, img1, img2):
		if self.img_info_dict[img1]['category'] == self.img_info_dict[img2]['category']:
			return True
		else:
			return False

	def common_group(self, img1, img2):
		groups_for_img1 = self.img_group_url_dict[img1]
		groups_for_img2 = self.img_group_url_dict[img2]
		common_group = []
		for group in groups_for_img1:
			if group in groups_for_img2:
				common_group.append(group)
		return common_group

	def get_likes_count(self, img, group):
		return float(self.img_info_dict[img][group]['Likes'])

	def get_views_count(self, img, group):
		return float(self.img_info_dict[img][group]['Meta views'])

	def get_time_elapsed(self, img, group):
		return float(self.img_info_dict[img][group]['Time elapsed'])

	def matching_based_on_views_vs_time_percentage(self, img, group, cutoff_percentage):
		likes = self.get_likes_count(img, group)
		views = self.get_views_count(img, group)
		time_elapsed = self.get_time_elapsed(img, group)*self.views_per_day

		return False if (views / time_elapsed) * 100.0 < cutoff_percentage else True

	def matching_based_on_likes_vs_views_percentage(self, img, group, cutoff_percentage):
		likes = self.get_likes_count(img, group)
		views = self.get_views_count(img, group)
		time_elapsed = self.get_time_elapsed(img, group)

		return False if (likes / views) * 100.0 < cutoff_percentage else True

	def matching_based_on_likes_vs_time_percentage(self, img, group, cutoff_percentage):
		likes = self.get_likes_count(img, group)
		views = self.get_time_elapsed(img, group)
		time_elapsed = self.get_time_elapsed(img, group)*self.likes_per_day

		return False if (likes / time_elapsed) * 100.0 < cutoff_percentage else True

	def get_average_likes_views_percentage(self):
		percentages = []
		for img in self.unique_image_urls:
			for group in self.img_group_url_dict[img]:
				percent = self.get_likes_count(img, group) / self.get_views_count(img, group) * 100.0
				percent = 100.0 if percent > 100.0 else percent
				percentages.append(percent)
		arr_len = len(percentages)
		self.average_lv = sum(percentages) / arr_len
		return percentages

	def get_average_views_time_percentage(self):
		percentages = []
		for img in self.unique_image_urls:
			for group in self.img_group_url_dict[img]:
				percent = self.get_views_count(img, group) / (self.get_time_elapsed(img, group)*self.views_per_day) * 100.0
				percent = 100.0 if percent > 100.0 else percent
				percentages.append(percent)
		arr_len = len(percentages)
		self.average_vt = sum(percentages) / arr_len
		return 	percentages

	def get_average_likes_time_percentage(self):
		percentages = []
		for img in self.unique_image_urls:
			for group in self.img_group_url_dict[img]:
				percent = self.get_likes_count(img, group) / (self.get_time_elapsed(img, group)*self.likes_per_day) * 100.0
				percent = 100.0 if percent > 100.0 else percent
				percentages.append(percent)
		arr_len = len(percentages)
		self.average_lt = sum(percentages) / arr_len
		return percentages

	def calculate_averages(self):
		self.get_average_likes_time_percentage()
		self.get_average_views_time_percentage()
		self.get_average_likes_views_percentage()

	def matching_percentage(self, img1, img2):
		matching_percent = 0.0
		if self.if_in_same_group(img1, img2) == True and self.if_in_same_category(img1, img2) == False:
			#print('Within same group')
			matching_percent = 50.0
			common_group = self.common_group(img1, img2)
			group_count = len(common_group) * 3.0
			group_matches = 0.0
			for group in common_group:
				if self.matching_based_on_views_vs_time_percentage(img1, group, self.average_vt):
					#print('Views vs time ok')
					group_matches += 1.0
				if self.matching_based_on_likes_vs_views_percentage(img1, group, self.average_lv):
					#print('Likes vs views ok')
					group_matches += 1.0
				if self.matching_based_on_likes_vs_time_percentage(img1, group, self.average_lt):
					#print('Likes vs time ok')
					group_matches += 1.0

			matching_percent += (group_matches / group_count) * 50.0

		else:
			matching_percent = 0.0

		return matching_percent

	# Discrete matching output, 0 for mismatch, 1 for match
	def matching_discrete(self, img1, img2):
		if self.if_in_same_group(img1, img2) == True:
			return 1.0
		else:
			return 0.0

	# Generate matches depending on the size of training set, return a list of tuples for matches and mismatches
	def generate_matches(self, bound):
		matches = []
		mismatches = []
		random.shuffle(self.unique_image_urls)
		for img1 in self.unique_image_urls[:bound]:
			for img2 in self.unique_image_urls[:bound]:
				if img1 != img2:
					percentage = self.matching_percentage(img1, img2)
					img1path = self.img_info_dict[img1]['path']
					img2path = self.img_info_dict[img2]['path']
					img1category= self.img_info_dict[img1]['category']
					img2category= self.img_info_dict[img2]['category']
					if percentage < 50.0:
						mismatches.append((img1path, img2path, img1category, img2category, 0, percentage))
					else:
						matches.append((img1path, img2path, img1category, img2category, 1, percentage))

		return matches, mismatches, matches + mismatches
