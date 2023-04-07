#task 1
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize']=(12,8)

#task 2
df=pd.read_csv("test.csv")
print(df.info())
# print(df)

#task 3
ax=sns.scatterplot(data=df,x="x_coordinate",y="y_coordinate")
ax.set_title("X coordinate vs Y coordinate")
plt.show()

#task 4
def cost_fun(x,y,theta):
    m=len(y)
    y_=x.dot(theta)
    error=(y_-y)**2
    return np.sum(error)/(2*m)
m=df.x_coordinate.values.size
x=np.append(np.ones((m,1)),df.x_coordinate.values.reshape(m,1),axis=1)
y=df.y_coordinate.values.reshape(m,1)
theta=np.zeros((2,1))
print(cost_fun(x,y,theta))

#task 5
def gradient_descent(x,y,theta,alpha,iteration):
    m=len(y)
    cost=[]
    for i in range(iteration):
        y_=x.dot(theta)
        error=np.dot(x.transpose(),(y_-y))
        # error=x.transpose().dot((y_-y))
        theta=theta-alpha*error/m
        cost.append(cost_fun(x,y,theta))   
    return theta,cost
theta,cost=gradient_descent(x,y,theta,0.00003,50)
print(f"h(x)={theta[0][0]} + {theta[1][0]} x1")

#task 6
from mpl_toolkits.mplot3d import Axes3D
theta_0=np.linspace(-10,10,100)
theta_1=np.linspace(-1,4,100)
cost_values=np.zeros((len(theta_0),len(theta_1)))
for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t=np.array([theta_0[i],theta_1[j]])
        cost_values[i,j]=cost_fun(x,y,t)
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(projection='3d')
surf=ax.plot_surface(theta_0,theta_1,cost_values,cmap=('viridis'))
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.xlabel("$\Theta_0$") 
plt.ylabel("$\Theta_1$")   
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30,330)   
plt.show()

#task 7     
plt.plot(cost)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Values of cost function over iterations in gradient descent")
plt.show()

#task 8
theta.shape
theta=np.squeeze(theta)
sns.scatterplot(x="x_coordinate",y="y_coordinate",data=df)
x_val=[x for x in range(5,100)]
y_val=[(x*theta[1]+theta[0]) for x in x_val]
sns.lineplot(x=x_val,y=y_val,color='green')
plt.xlabel("x_coordinate")
plt.ylabel("y_coordinate")
plt.title("Linear regerssion fit")
plt.show()

n=int(input('Enter x coordinate value: '))
def prediction(x,theta):
    y_=x.dot(theta)
    return y_

y_pred1=prediction(np.array([1,n]),theta)
print("For X= ",n,end=" ")
print("Y= "+str(round(y_pred1,3)))