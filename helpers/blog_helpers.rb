require 'redcarpet'

module BlogHelpers
  def markdown(contents)
    renderer = Redcarpet::Render::HTML
    markdown = Redcarpet::Markdown.new(
      renderer,
      autolink: true,
      fenced_code_blocks: true,
      footnotes: true,
      highlight: true,
      smartypants: true,
      strikethrough: true,
      tables: true,
      with_toc_data: true
    )
    markdown.render(contents)
  end
  
  def page_title
    title = "YNorm"

    if current_page.data.title
      title = current_page.data.title + " | YNorm"
    end

    title
  end

  def formatted_title(title)
    markdown = Redcarpet::Markdown.new(Redcarpet::Render::HTML, autolink: true)
    rendered_title = markdown.render title
    Regexp.new('^<p>(.*)<\/p>$').match(rendered_title)[1]
  end
end
