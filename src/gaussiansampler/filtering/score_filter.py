class InteractiveScoreFilter:
  def __init__(self, scorer, interactor):
    self._scorer = scorer
    self._interactor = interactor

  def filter(self, objects):
    scores = self._scorer.score(objects)
    left = 0
    right = len(objects)
    while left < right:
      m = (left + right) // 2
      if self._interactor.ask(scores[m]):
        right = m
      else:
        left = m + 1
    return [path for path, score in scores[left:]]

def filter_by_threshold(objects, scorer, threshold):
  scores = scorer.score(objects)
  return [obj for obj, score in scores if score >= threshold]