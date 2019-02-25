---
layout: default
---

Learning to Place (L2P) is an algorithm designed to predict heavy-tail distributed outcomes. This methodology is currently under review.

Many real-world prediction tasks have outcome (a.k.a., target) variables that have characteristic heavy-tail distributions. Examples include copies of books sold, auction prices of art pieces, and sales of movies in the box office. Accurate predictions for the "big and rare" instances (e.g., the best-sellers, the box-office hits, etc) is a hard task. Most existing approaches heavily under-predict such instances because they cannot deal effectively with heavy-tailed distributions. We introduce Learning to Place (L2P), which exploits the pairwise relationships between instances to learn from a proportionally higher number of rare instances. L2P consists of two phases. In Phase 1, L2P learns a pairwise preference classifier: is instance A > instance B?. In Phase 2, L2P learns to place an instance from the output of Phase 1. Based on its placement, the instance is then assigned a value for its outcome variable. Our experiments, on real-world and synthetic datasets, show that our L2P approach outperforms competing approaches and provides explainable outcomes.

## Algorithm

![Branching]({{site.baseurl}}/assets/img/flowchart_Ltp.png)> 

## Usage

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;

```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
