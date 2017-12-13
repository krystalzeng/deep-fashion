import numpy as  np

def if_in_same_group(img1, img2):
	groups_for_img1 = img_group_url_dict[img1]
	groups_for_img2 = img_group_url_dict[img2]
	for group in groups_for_img1:
		if group in groups_for_img2:
			return True

	for group in groups_for_img2:
		if group in groups_for_img1:
			return True

	return False

def get_likes_vs_views_percentage(img, group):
	if img in img_info_dict:
		if group in img_info_dict[img]:
			return img_info_dict[img][group]['Likes vs. Views (%)']
	return 0.0

def get_views_vs_time_percentage(img, group):
	if img in img_info_dict:
		if group in img_info_dict[img]:
			return img_info_dict[img][group]['Views vs. Time elapsed (%)']
	return 0.0

def matching_based_on_views_vs_time_percentage(img, group, cutoff_percentage):
	return False if get_views_vs_time_percentage(img, group) < cutoff_percentage else True


def matching_based_on_likes_vs_views_percentage(img, group, cutoff_percentage):
	return False if get_likes_vs_views_percentage(img, group) < cutoff_percentage else True

def matching(img, group):
	return matching_based_on_views_vs_time_percentage(duplicate_url, duplicate_url_group_id, 50) and \
		matching_based_on_likes_vs_views_percentage(duplicate_url, duplicate_url_group_id, 50)