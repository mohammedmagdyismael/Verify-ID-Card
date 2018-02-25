# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:57:53 2018

@author: mohammed-PC
"""
from difflib import SequenceMatcher


def longestSubstring(str1,str2):
     seqMatch = SequenceMatcher(None,str1,str2) 
     match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
#     if (match.size!=0):
#          print (str1[match.a: match.a + match.size]) 
     return str1[match.a: match.a + match.size]