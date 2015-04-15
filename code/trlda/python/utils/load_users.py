try:
	from numpy.random import poisson
except:
	pass

from collections import defaultdict

def load_users(filepath, batch_size=None, stochastic=False, threshold=4):
	"""
	Load users from a text file. If `batch_size` is given, behaves like a generator
	and returns one batch at a time. Each user is represented as a list of tuples,
	where each tuple contains an item id and a rating.

	Each line is assumed to contain a 3-tuple of a user id, an item id, and a rating.
	The ratings of users should be grouped. For example::

		1488844   1  3
		1488844   8  4
		1488844  17  2
		1488844  30  3
		8850131  33  4
		8850131  35  1
		8850131  86  5

	@type  filepath: C{str}
	@param filepath: path to file containing data

	@type  batch_size: C{int}
	@param batch_size: the number of users to return at once

	@type  stochastic: C{bool}
	@param stochastic: if True, the batch size is drawn from a Poisson distribution

	@type  threshold: C{int}
	@param threshold: only load users whose rating is greater or equal this threshold

	@rtype: C{list}/C{generator}
	@return: returns either a list of users or a generator of lists of users

	@seealso: L{load_documents()}
	"""

	def user_generator(filepath, batch_size):
		user = []
		users = []
		current_uid = None

		# draw a possibly random number of users
		current_batch_size = poisson(batch_size) if stochastic else batch_size

		with open(filepath) as handle:
			for line in handle:
				# load single user/item pair
				uid, item, rating = (int(i) for i in line.split())

				if threshold > 0:
					if rating < threshold:
						# skip item
						continue
					rating = 1

				if uid != current_uid:
					# end of user detected
					if user:
						users.append(user)

						if batch_size:
							while current_batch_size == 0:
								yield []
								current_batch_size = poisson(batch_size)

							if len(users) >= current_batch_size:
								# return batch of users
								yield users
								users = []

							if stochastic:
								# draw a new random batch size
								current_batch_size = poisson(batch_size)

					# start new user
					user = []
					current_uid = uid

				# add item to user
				user.append((item, rating))

			if user:
				users.append(user)
				user = []

		yield users

	if batch_size:
		return user_generator(filepath, batch_size)
	return next(user_generator(filepath, batch_size))



def load_users_as_dict(filepath, batch_size=None, stochastic=False, threshold=4):
	"""
	Like L{load_users}, but users are stored in a dictionary instead of a list.
	The keys correspond to user IDs and the values are lists of item/rating pairs.

	@rtype: C{dict}/C{generator}
	@return: returns either a dictionary of users or a generator of dictionaries of users
	"""

	def user_generator(filepath, batch_size):
		user = []
		users = {}
		current_uid = None

		# draw a possibly random number of users
		current_batch_size = poisson(batch_size) if stochastic else batch_size

		with open(filepath) as handle:
			for line in handle:
				# load single user/item pair
				uid, item, rating = (int(i) for i in line.split())

				if threshold > 0:
					if rating < threshold:
						# skip item
						continue
					rating = 1

				if uid != current_uid:
					# end of user detected
					if user:
						users[current_uid] = user

						if batch_size:
							while current_batch_size == 0:
								yield []
								current_batch_size = poisson(batch_size)

							if len(users) >= current_batch_size:
								# return batch of users
								yield users
								users = {}

							if stochastic:
								# draw a new random batch size
								current_batch_size = poisson(batch_size)

					# start new user
					user = []
					current_uid = uid

				# add item to user
				user.append((item, rating))

			if user:
				users[current_uid] = user
				user = []

		yield users

	if batch_size:
		return user_generator(filepath, batch_size)
	return next(user_generator(filepath, batch_size))
