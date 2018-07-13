class dataLoader:
	def __init__(self, x, batchSize):
		self.x = x
		self.dataSize = x.shape[0]
		self.count = 0
		self.maxLimit = (self.dataSize // batchSize) + 1 
		self.batchSize = batchSize

	def __iter__(self):
		return self

	def next(self):
		if self.count > self.maxLimit:
			raise StopIteration
		else:
			self.count += 1
			return self.x[self.batchSize * (self.count-1): self.batchSize * self.count]


