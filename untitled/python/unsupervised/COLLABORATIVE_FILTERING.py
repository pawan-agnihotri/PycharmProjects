data = {"Jacob": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 3.5, "Edge of Tomorrow": 5.0, "The Maze Runner": 3.0},
     "Jane": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 4.0, "Edge of Tomorrow": 3.0, "The Maze Runner": 3.0, "Unbroken": 4.5},
     "Jim": {"The Martin": 1.0, "Guardians of the Galaxy": 2.0, "Edge of Tomorrow": 5.0, "The Maze Runner": 4.5, "Unbroken": 2.0}}
print (data.get("Jacob"))
#We can also find the movies that have been watched by Jacob and Jane by
# using the intersection of both sets.
# We will save this as a list in common_movies for later use.
common_movies = list(set(data.get("Jacob")).intersection(data.get("Jane")))
print (common_movies)
#Similarly, we can find possible recommendations by finding the movies that Jane has watched
# and the ones Jacob has not. This can be acomplished by using the difference function.
# Some recommendation systems will use Collaborative Filtering to looking for people that rate movies similar to pull recommendations from their watched list.
recommendation = list(set(data.get("Jane")).difference(data.get("Jacob")))
print(recommendation)

#similar_mean function to compute the average difference in rating.
# This will tell if ratings on the movies are similar or not.
# We'll have threshold of 1 rating or less to consider the recommendation an adequate one.
def similar_mean(same_movies, user1, user2, dataset):
    total = 0
    for movie in same_movies:
        total += abs(dataset.get(user1).get(movie) - dataset.get(user2).get(movie))
    return total/len(same_movies)
print(similar_mean(common_movies, "Jacob", "Jane", data))

# lets recommand for Jim

common_movies1 = list(set(data.get("Jacob")).intersection(data.get("Jim")))
recommendation1 = list(set(data.get("Jacob")).difference(data.get("Jim")))
print (common_movies1)
print(recommendation1)
print(similar_mean(common_movies1, "Jacob", "Jim", data))
print ("Since the average difference in rating is larger than our 1.0 threshold, 'The Avengers' would not be a good recommendation for Jim.")
