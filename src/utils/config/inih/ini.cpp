/* inih -- simple .INI file parser

inih is released under the New BSD license (see LICENSE.txt). Go to the project
home page for more info:

http://code.google.com/p/inih/

*/

#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include <sstream>
#include <iostream>
#include <string>
#include <cctype> // for std::isspace

#include "ini.h"

#define MAX_LINE 200
#define MAX_SECTION 50
#define MAX_NAME 50

/* Strip whitespace chars off end of given string, in place. Return s. */
static char *
rstrip(char * s)
{
  char * p = s + strlen(s);
  while (p > s && isspace((unsigned char)(*--p)))
    *p = '\0';
  return s;
}

/* Return pointer to first non-whitespace char in given string. */
static char *
lskip(const char * s)
{
  while (*s && isspace((unsigned char)(*s)))
    s++;
  return (char *)s;
}

/* Return pointer to first char c or ';' ( or '#') comment in given
 string, or pointer to null at end of string if neither found. ';' (or
 '#') must be prefixed by a whitespace character to register as a comment. */
static char *
find_char_or_comment(const char * s, char c)
{
  int was_whitespace = 0;
  while (*s && *s != c && !(was_whitespace && *s == ';'))
  {
    was_whitespace = isspace(*s);
    s++;
  }
  return (char *)s;
}

/* Version of strncpy that ensures dest (size bytes) is null-terminated. */
static char *
strncpy0(char * dest, const char * src, size_t size)
{
  /* Could use strncpy internally, but it causes gcc warnings */
  size_t i;
  for (i = 0; i < size - 1 && src[i]; i++)
    dest[i] = src[i];
  dest[i] = '\0';
  return dest;
}

// ================================================================
// ================================================================
// ================================================================
/* See documentation in header file. */
int
ini_parse(const char * filename,
          int (*handler)(void *, const char *, const char *, const char *),
          void * user)
{
  /* Uses a fair bit of stack (use heap instead if you need to) */
  char line[MAX_LINE];
  char section[MAX_SECTION] = "";
  char prev_name[MAX_NAME] = "";

  FILE * file;
  char * start;
  char * end;
  char * name;
  char * value;
  int    lineno = 0;
  int    error = 0;

  file = fopen(filename, "r");
  if (!file)
    return -1;

  /* Scan through file line by line */
  while (fgets(line, sizeof(line), file) != NULL)
  {
    lineno++;
    start = lskip(rstrip(line));

#if INI_ALLOW_MULTILINE
    if (*prev_name && *start && start > line)
    {
      /* Non-black line with leading whitespace, treat as continuation
         of previous name's value (as per Python ConfigParser). */
      if (!handler(user, section, prev_name, start) && !error)
        error = lineno;
    }
    else
#endif
      if (*start == ';' || *start == '#')
    {
      /* Per Python ConfigParser, allow '#' comments at start of line */
    }
    else if (*start == '[')
    {
      /* A "[section]" line */
      end = find_char_or_comment(start + 1, ']');
      if (*end == ']')
      {
        *end = '\0';
        strncpy0(section, start + 1, sizeof(section));
        *prev_name = '\0';
      }
      else if (!error)
      {
        /* No ']' found on section line */
        error = lineno;
      }
    }
    else if (*start && *start != ';')
    {
      /* Not a comment, must be a name=value pair */
      end = find_char_or_comment(start, '=');
      if (*end == '=')
      {
        *end = '\0';
        name = rstrip(start);
        value = lskip(end + 1);
        end = find_char_or_comment(value, '\0');
        if (*end == ';')
          *end = '\0';
        rstrip(value);

        /* Valid name=value pair found, call handler */
        strncpy0(prev_name, name, sizeof(prev_name));
        if (!handler(user, section, name, value) && !error)
          error = lineno;
      }
      else if (!error)
      {
        /* No '=' found on name=value line */
        error = lineno;
      }
    }
  }

  fclose(file);

  return error;

} // ini_parse

// ================================================================
// ================================================================
// ================================================================
int
ini_parse_buffer(char *& buffer,
                 int     buffer_size,
                 int (*handler)(void *, const char *, const char *, const char *),
                 void * user)
{
  /* Uses a fair bit of stack (use heap instead if you need to) */
  char * line = NULL;
  char   section[MAX_SECTION] = "";
  char   prev_name[MAX_NAME] = "";

  // convert the input buffer into an input string stream
  std::istringstream strs;
  strs.rdbuf()->pubsetbuf(buffer, buffer_size);


  char * start;
  char * end;
  char * name;
  char * value;
  int    lineno = 0;
  int    error = 0;

  // Scan through string stream line by line
  for (std::string linestr = ""; std::getline(strs, linestr);)
  {

    lineno++;

    // copy std::string into a regular C char array
    line = new char[linestr.length() + 1];
    strcpy(line, linestr.c_str());

    start = lskip(rstrip(line));

#if INI_ALLOW_MULTILINE
    if (*prev_name && *start && start > line)
    {
      /* Non-black line with leading whitespace, treat as continuation
   of previous name's value (as per Python ConfigParser). */
      if (!handler(user, section, prev_name, start) && !error)
        error = lineno;
    }
    else
#endif
      if (*start == ';' || *start == '#')
    {
      /* Per Python ConfigParser, allow '#' comments at start of line */
    }
    else if (*start == '[')
    {
      /* A "[section]" line */
      end = find_char_or_comment(start + 1, ']');
      if (*end == ']')
      {
        *end = '\0';
        strncpy0(section, start + 1, sizeof(section));
        *prev_name = '\0';
      }
      else if (!error)
      {
        /* No ']' found on section line */
        error = lineno;
      }
    }
    else if (*start && *start != ';')
    {
      /* Not a comment, must be a name=value pair */
      end = find_char_or_comment(start, '=');
      if (*end == '=')
      {
        *end = '\0';
        name = rstrip(start);
        value = lskip(end + 1);
        end = find_char_or_comment(value, '\0');
        if (*end == ';')
          *end = '\0';
        rstrip(value);

        /* Valid name=value pair found, call handler */
        strncpy0(prev_name, name, sizeof(prev_name));
        if (!handler(user, section, name, value) && !error)
          error = lineno;
      }
      else if (!error)
      {
        /* No '=' found on name=value line */
        error = lineno;
      }
    }

    delete[] line;
  }

  return error;

} // init_parse_buffer
