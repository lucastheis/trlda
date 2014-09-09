try:
	from numpy.random import poisson
except:
	pass

def load_users(filepath, batch_size=None, stochastic=False):
	"""
	Load users from a text file. If `batch_size` is given, behaves like a generator
	and returns one batch at a time. Each user is represented as a list of tuples,
	where each tuple contains an item id and a rating.

	@type  batch_size: C{int}
	@param batch_size: the number of users to return at once

	@type  stochastic: C{bool}
	@param stochastic: if True, the batch size is drawn from a Poisson distribution

	@rtype: C{list}/C{generator}
	@return: returns either a list of users or a generator of lists of users
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
