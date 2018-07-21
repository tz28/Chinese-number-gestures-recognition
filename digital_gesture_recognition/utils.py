from PIL import  Image
import os


def split_file():
	dirs = os.listdir("generater_pic")
	counter = 1
	index = 0
	for filename in dirs:
		if counter == 1:
			os.mkdir("split_pic//{}".format(str(index)))
		im = Image.open("generater_pic//{}".format(filename))
		im.save("split_pic//{}//{}".format(str(index), filename))
		counter += 1
		if counter == 2001:
			counter = 1
			index += 1

if __name__ == "__main__":
	split_file()

