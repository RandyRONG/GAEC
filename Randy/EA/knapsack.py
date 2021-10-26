
def track(d, c, w):
    # d是dp数组
    # c是背包容量
    # w是物品的数据
    x = []
    for i in range(len(w), 1, -1):
        if d[i][c] != d[i-1][c]:
            x.append(w[i-1])
            c = c - w[i-1]

    # 拿一个物品时，价值最大的这个值肯定是需要的
    if d[1][c] > 0:
        x.append(w[0])

    return x

if __name__ == "__main__":
    # c = int(input())
    c = 12
    w = [4,6,2,2,5,1]
    v = [8,10,6,3,7,2]
    n = len(w)
    # 置零，表示初始状态
    dp = [[0 for j in range(c+1)] for i in range(n + 1)]
    for i in range(1, n+1):
        for j in range(1, c+1):
            dp[i][j] = dp[i-1][j]
            # 背包总容量能够放当前物品，遍历前一个状态考虑是否置换
            if j >= w[i-1] and dp[i][j] < dp[i-1][j-w[i-1]] + v[i-1]:
                dp[i][j] = max(dp[i-1][j-w[i-1]] + v[i-1], dp[i-1][j])

    for x in dp:
        print(x)

    print(track(dp,c,w))

# 输入
# 12
# 4,6,2,2,5,1
# 8,10,6,3,7,2