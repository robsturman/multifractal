import numpy as np
import matplotlib.pyplot as plt

def segment_length(xl,xr,yb,yt):
    return np.sqrt((xr-xl)**2+(yt-yb)**2)

def segments_total(starts,ends):
    total_length = 0.0
    for j in range(0,len(starts)):
        total_length += segment_length(starts[j][0],ends[j][0],starts[j][1],ends[j][1])
    return total_length

# Definition of D_q, for some small grid box (1/n)
def Dq(p_i,q,n):
    # L'Hopital for the q=1 case
    if q==1:
        zq = 0.0
        for i in range(n):
            for j in range(n):
                if p_i[i][j]>0: ### need to check this
                    zq += p_i[i][j]*np.log(p_i[i][j])
        return zq/(np.log(1/n))
    else:
        zq = 0
        for i in range(n):
            for j in range(n):
                if p_i[i][j]>0: ### need to check this line!
                    zq += p_i[i][j]**q
        return np.log(zq)/(-np.log(1/n)*(1-q))

# Definition of D_q, for some small grid box (1/n)
def Dq_1d(p_i,q,n):
    # L'Hopital for the q=1 case
    if q==1:
        zq = 0.0
        for i in range(n):
            zq += p_i[i]*np.log(p_i[i])
        return zq/(np.log(1/n))
    else:
        zq = 0
        for i in range(n):
                if p_i[i]>0: ### need to check this line!
                    zq += p_i[i]**q
        return np.log(zq)/(-np.log(1/n)*(1-q))
    
def mu_i(p_i,q,n):
    mui = np.zeros([n,n])
    tot = 0.0
    for i in range(n):
        for j in range(n):
            if p_i[i][j]>0:
                mui[i][j] = p_i[i][j]**q
                tot += p_i[i][j]**q
    return mui/tot

def mu_i_1d(p_i,q,n):
    mui = np.zeros(n)
    tot = 0.0
    for i in range(n):
            if p_i[i]>0:
                mui[i] = p_i[i]**q
                tot += p_i[i]**q
    return mui/tot

def f_of_q(mui,q,n):
    tot = 0
    for i in range(n):
        for j in range(n):
            if mui[i][j]>0:
                tot += mui[i][j]*np.log(mui[i][j])
    return tot/np.log(1/n)

def f_of_q_1d(mui,q,n):
    tot = 0
    for i in range(n):
            if mui[i]>0:
                tot += mui[i]*np.log(mui[i])
    return tot/np.log(1/n)

def alpha_of_q(mui,p_i,q,n):
    tot = 0
    for i in range(n):
        for j in range(n):
            if p_i[i][j]>0:
                tot += mui[i][j]*np.log(p_i[i][j])
    return tot/np.log(1/n)

def alpha_of_q_1d(mui,p_i,q,n):
    tot = 0
    for i in range(n):
            if p_i[i]>0:
                tot += mui[i]*np.log(p_i[i])
    return tot/np.log(1/n)

# plot the generalised dimensions
def plot_Dq(gridprops,n):        
    fig, ax = plt.subplots()
    plt.xlabel('q')
    plt.ylabel('Dq')
    dqs = []
    for qqq in np.arange(-20,20,0.5):
        dqs.append(Dq(gridprops,qqq,n))
    ax.plot(np.arange(-20,20,0.5),dqs)
    return None

# plot the generalised dimensions
def plot_Dq_1d(gridprops,n):        
    fig, ax = plt.subplots()
    dqs = []
    for qqq in np.arange(-20,20,0.5):
        dqs.append(Dq_1d(gridprops,qqq,n))
    ax.plot(np.arange(-20,20,0.5),dqs)
    return None

def range_of_Dq(gridprops,n):
    dqs = []
    for qqq in np.arange(-20,20,0.5):
        dqs.append(Dq(gridprops,qqq,n))
    return np.max(dqs)-np.min(dqs)


def f(x):
    return x**3

def intersections_of_line_with_line(x0,y0,x1,y1,x2,y2,x3,y3):
    # here we assume the line y = 1/4 as in Giona2005
    # x2, y2, x3, y3 = 0, 1/4, 1, 1/4
    if ((x0-x1)*(y2-y3)-(y0-y1)*(x2-x3))==0.0:
       return 0,0
    t = ((x0-x2)*(y2-y3)-(y0-y2)*(x2-x3))/((x0-x1)*(y2-y3)-(y0-y1)*(x2-x3))
    u = -((x0-x1)*(y0-y2)-(y0-y1)*(x0-x2))/((x0-x1)*(y2-y3)-(y0-y1)*(x2-x3))
    
    if t>=0.0 and t<=1.0:
        return x0+t*(x1-x0), y0+t*(y1-y0)
    if u>=0.0 and u<=0.0:
        return x2+u*(x3-x2), y2+u*(y3-y2)
    else:
        return 2.0,0.25
    
    
def intersection_measure(starts,ends,x2,y2,x3,y3,n):
    # here we assume the line y = 1/4 as in Giona2005
    # x2, y2, x3, y3 = 0, 0, 1, 0
    interval_props = np.zeros(n)
    for i in range(len(starts)):
        intersection, b = intersections_of_line_with_line(starts[i][0],starts[i][1],ends[i][0],ends[i][1],x2,y2,x3,y3)
        if intersection<1.5:
            interval_props[int(intersection*n)] +=1
    return interval_props/np.sum(interval_props)

# Here we find the intersections of line segments with a
# grid of boxes.
# We assume an n x n grid  in the unit square
# A line segment can start and finish anywhere not outside
# the unit square

def intersections_of_line_with_grid(x0,y0,x1,y1,n):
    
    # gradient of the line segment 
    phi = (y1-y0)/(x1-x0)
    
    # width of a box
    res = 1.0/n
    # which box do we start in?
    i, j = int(x0*n), int(y0*n)
    # which box do we end in?
    ii, jj = int(x1*n), int(y1*n)
    # if we're on a gridline, we want the box to the
    # left or underneath
    if i == n:
        i -= 1
    if j ==n:
        j -= 1
    if ii == n:
        ii -= 1
    if jj ==n:
        jj -= 1
    
    # record the grid intersections
    inters = []
    inters.append((x0,y0))
    # if line segment entirely in a box, just add the endpoint
    # and exit - there are no more intersections
    if i==ii and j==jj:
        inters.append((x1,y1))
        return inters
    
    # otherwise step onto the vertical gridlines
    # unless we don't cross any
    if i!=ii:
        dx = (i+1)*res-x0
        dy = phi*dx
        x, y = x0+dx, y0+dy
        inters.append((x,y))
        
        # iterate across the other available vertical gridlines
        for k in range(ii-i-1):
            dx, dy = res, phi*res
            x, y = x+dx, y+dy
            inters.append((x,y))

    # back to (x0,y0) and step onto the horizontal gridlines
    # unless we don't cross any
    if j!=jj:
        dy = (j+1)*res-y0
        dx = dy/phi
        x, y = x0+dx, y0+dy
        inters.append((x,y))
    
        # iterate across the other available horizontal gridlines
        for k in range(jj-j-1):
            dx, dy = res/phi, res
            x, y = x+dx, y+dy
            inters.append((x,y))
        
    # finish with the last endpoint
    # unless it's on a gridline
    if inters[-1]!=(x1,y1):
        inters.append((x1,y1))
    # sort the intersection  points by x-value
    inters.sort()
    
    return inters


# Now we want to get the lengths of each segment
# in each grid box, which we do by finding
# grid intersections and using segment_length
def length_of_segment_in_box(starts,ends,n):
    # matrix to store lengths in each gridbox
    grid_props = np.zeros((n,n))

    for j in range(0,len(starts)):
        # if the line goes left to right
        if starts[j][0]<ends[j][0]:
            sub_ends = intersections_of_line_with_grid(starts[j][0],starts[j][1],ends[j][0],ends[j][1],n)
            for i in range(len(sub_ends)-1):
                length = segment_length(sub_ends[i][0],sub_ends[i+1][0],sub_ends[i][1],sub_ends[i+1][1])
                #find which box the subsegment is in by looking
                # at its midpoint
                ii, jj = int(n*(sub_ends[i][0]+sub_ends[i+1][0])/2), int(n*(sub_ends[i][1]+sub_ends[i+1][1])/2)
                if ii<n and jj<n:
                    grid_props[ii][jj] +=length
        # if the line goes right to left
        if starts[j][0]>ends[j][0]:
            sub_ends = intersections_of_line_with_grid(ends[j][0],ends[j][1],starts[j][0],starts[j][1],n)
            for i in range(len(sub_ends)-1):
                length = segment_length(sub_ends[i][0],sub_ends[i+1][0],sub_ends[i][1],sub_ends[i+1][1])
                ii, jj = int(n*(sub_ends[i][0]+sub_ends[i+1][0])/2), int(n*(sub_ends[i][1]+sub_ends[i+1][1])/2)
                if ii<n and jj<n:
                    grid_props[ii][jj] += length  
    grid_props = grid_props/segments_total(starts,ends)
    return grid_props

                                                                                                                        