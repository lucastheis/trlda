try:
	from numpy.random import poisson
except:
	pass

def load_documents(filepath, batch_size=None, stochastic=False):
	"""
	Load documents from a text file. If C{batch_size} is given, behaves like a generator
	and returns one batch at a time. Each document is represented as a list of tuples,
	where each tuple contains a word id and a word count.

	Each line of the text file is assumed to contain one document and should start with
	the number of unique words in that document, followed by the words. Each word
	should be represented by its id and a number of occurences separated by a colon. For
	example::

		6 5600:2 293:1 5548:1 2577:1 3733:3 2677:2

	@type  batch_size: C{int}
	@param batch_size: the number of documents to return at once

	@type  stochastic: C{bool}
	@param stochastic: if True, the batch size is drawn from a Poisson distribution

	@rtype: C{list}/C{generator}
	@return: returns either a list of documents or a generator of lists of documents

	@seealso: L{load_users()}
	"""

	def document_generator(filepath, batch_size):
		# load documents from file
		documents = []

		# draw a possibly random number of documents
		current_batch_size = poisson(batch_size) if stochastic else batch_size

		with open(filepath) as handle:
			# each line is a document
			for lineno, line in enumerate(handle):
				document = []

				for word in line.split()[1:]:
					wid, wct = word.split(':')
					document.append((int(wid), int(wct)))

				documents.append(document)

				if batch_size:
					while current_batch_size == 0:
						yield []
						current_batch_size = poisson(batch_size)

					if (lineno + 1) % current_batch_size == 0:
						# batch is full, return documents
						yield documents
						documents = []

						if stochastic:
							# draw a new random batch size
							current_batch_size = poisson(batch_size)

		yield documents

	if batch_size:
		# return generator
		return document_generator(filepath, batch_size)
	# return all documents
	return next(document_generator(filepath, batch_size))
