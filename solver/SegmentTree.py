class SegmentTree:
    def __init__(self, nums):
        self.nums = nums
        self.tree = [0] * (4 * len(nums))
        self.build(0, 0, len(nums) - 1)

    def build(self, node, start, end):
        if start == end:
            self.tree[node] = self.nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self.build(left_child, start, mid)
            self.build(right_child, mid + 1, end)
            self.tree[node] = max(self.tree[left_child], self.tree[right_child])

    def query(self, node, start, end, L, R):
        if R < start or end < L:
            return float('-inf')
        if L <= start and end <= R:
            return self.tree[node]
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        left_max = self.query(left_child, start, mid, L, R)
        right_max = self.query(right_child, mid + 1, end, L, R)
        return max(left_max, right_max)