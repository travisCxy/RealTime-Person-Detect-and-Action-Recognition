train_list = open('train_list.txt', 'r')
valid_list = open('valid_list.txt', 'r')
new_train_list = open('new_train_list.txt', 'w')
new_valid_list = open('new_valid_list.txt', 'w')

def replace_label(old_list, new_list):
	for line in old_list:
		label = line.split(' ')[2].strip()
		if label == '0':
			new_list.write(line)
		if label == '1':
			new_list.write(line)
		if label == '7':
			new_list.write(line.replace(' 7', ' 2'))
		if label == '8':
			new_list.write(line.replace(' 8', ' 3'))
		if label == '25':
			new_list.write(line.replace(' 25', ' 4'))
		if label == '26':
			new_list.write(line.replace(' 26', ' 5'))
		if label == '40':
			new_list.write(line.replace(' 40', ' 6'))
		if label == '41':
			new_list.write(line.replace(' 41', ' 7'))
		if label == '42':
			new_list.write(line.replace(' 42', ' 8'))
		if label == '43':
			new_list.write(line.replace(' 43', ' 9'))
		if label == '44':
			new_list.write(line.replace(' 44', ' 10'))
		if label == '45':
			new_list.write(line.replace(' 45', ' 11'))
		if label == '46':
			new_list.write(line.replace(' 46', ' 12'))
		if label == '47':
			new_list.write(line.replace(' 47', ' 13'))
		if label == '49':
			new_list.write(line.replace(' 49', ' 14'))
		if label == '50':
			new_list.write(line.replace(' 50', ' 15'))
		if label == '51':
			new_list.write(line.replace(' 51', ' 16'))
replace_label(train_list, new_train_list)
replace_label(valid_list, new_valid_list)